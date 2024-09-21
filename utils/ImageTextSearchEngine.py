import faiss, mysql.connector, clip, torch, requests, numpy as np
from PIL import Image
from langdetect import detect
from io import BytesIO


class ImageTextSearchEngine:
    # Initial search engine
    def __init__(self, db_config, bin_file, translator, text_preprocessing, clip_backbone='ViT-B/32', device='cpu'):
        self.db_connection = mysql.connector.connect(**db_config)
        self.db_cursor = self.db_connection.cursor()
        self.device = device
        self.model, self.preprocess = clip.load(clip_backbone, device=device)
        self.translator = translator
        self.text_preprocessing = text_preprocessing
        self.index = self.load_faiss_index(bin_file)

    def load_faiss_index(self, bin_file):
        if bin_file:
            return faiss.read_index(bin_file)
        return None
    
    def get_image_feature_by_tuple(self, id_tuple: tuple):
        placeholders = ','.join(['%s'] * len(id_tuple))
        order_by_case = " ".join([f"WHEN %s THEN {i}" for i in range(len(id_tuple))])
        sql = f"""
            SELECT img_feat.id, img_feat.folder_id, img_feat.child_folder_id, img_feat.id_frame, img_feat.image_path, img_feat.frame_mapping_index
            FROM image_features AS img_feat
            INNER JOIN (
                SELECT folder_id, child_folder_id, id_frame
                FROM image_features
            ) AS tmp
            ON tmp.folder_id = img_feat.folder_id
            AND tmp.child_folder_id = img_feat.child_folder_id
            AND tmp.id_frame = img_feat.id_frame
            AND img_feat.id IN ({placeholders})
            ORDER BY CASE img_feat.id {order_by_case} END;
        """
        parameters = id_tuple + id_tuple
        self.db_cursor.execute(sql, parameters)
        rows = self.db_cursor.fetchall()

        results = [{
            'id': row[0],
            'folder_id': row[1],
            'child_folder_id': row[2],
            'id_frame': row[3],
            'image_path': row[4],
            'frame_mapping_index': row[5]
        } for row in rows]

        return results
    
    # Search images by image
    def search_images_by_id(self, image_id, k):
        sql = "SELECT vector_features FROM image_features WHERE id = %s"
        self.db_cursor.execute(sql, (image_id,))
        row = self.db_cursor.fetchone()
        
        vector_blob = row[0]
        query_vector = np.frombuffer(vector_blob, dtype='float32')
        query_vector = query_vector.reshape(1, -1)

        _, indices = self.index.search(query_vector, k)

        indices = indices.flatten()

        if len(indices) == 0:
            return []
        
        id_tuple = tuple(int(i) for i in indices)

        return self.get_image_feature_by_tuple(id_tuple)

    # Search images by text
    def search_images_by_text(self, text:str, k):
        detect_lang = detect(text)

        if detect_lang == 'vi':
            translated_text = self.translator(text)
        else:
            translated_text = text
        
        text_preprocessed = self.text_preprocessing(translated_text)

        print(text_preprocessed)

        text_tokenized = clip.tokenize(text_preprocessed).to(self.device)

        with torch.no_grad():
            text_vector = self.model.encode_text(text_tokenized).cpu().numpy()
            distances, indices = self.index.search(text_vector, k)

            indices = indices.flatten()

            if len(indices) == 0:
                return []
            
            id_tuple = tuple(int(i) for i in indices)

            return self.get_image_feature_by_tuple(id_tuple)

    # Search images by a part of image
    def get_image_vector(self, image):
        with torch.no_grad():
            image_features = self.preprocess(image).unsqueeze(0).to(self.device)
            image_vector = self.model.encode_image(image_features).cpu().numpy()
            return image_vector

    def split_image(self, image_id, x1, x2, y1, y2):
        image_path = self.id2img_fps[image_id]
        image = self.load_image(image_path)

        return image.crop((x1, y1, x2, y2))

    def load_image(self, image_path):
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            if response.status_code != 200:
                raise ValueError(f'Unable to get image from {image_path}')
            return Image.open(BytesIO(response.content))
        return Image.open(image_path)

    def search_similar_images_for_part(self, imgId, x1, x2, y1, y2, k):
        part = self.split_image(image_id=imgId, x1=x1, x2=x2, y1=y1, y2=y2)
        part_vector = self.get_image_vector(part)

        with torch.no_grad():
            _, indices = self.index.search(part_vector, k)
            indices = indices.flatten()

            if len(indices) == 0:
                return []
            
            id_tuple = tuple(int(i) for i in indices)

            return self.get_image_feature_by_tuple(id_tuple)

    def close(self):
        self.db_cursor.close()
        self.db_connection.close()



