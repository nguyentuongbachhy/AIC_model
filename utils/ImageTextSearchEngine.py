import faiss, mysql.connector, torch, numpy as np, pandas as pd
from langdetect import detect
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer



class ImageTextSearchEngine:
    # Initial search engine
    def __init__(self, db_config, bin_file, translator, text_preprocessing, clip_backbone='ViT-B/32', device='cpu'):
        self.db_connection = mysql.connector.connect(**db_config)
        self.db_cursor = self.db_connection.cursor()
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.translator = translator
        self.text_preprocessing = text_preprocessing
        self.index = self.load_faiss_index(bin_file)

    def load_faiss_index(self, bin_file):
        if bin_file:
            return faiss.read_index(bin_file)
        return None

    def normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


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
        query_vector = self.normalize(query_vector)

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
        
        processed_text = self.text_preprocessing(translated_text)

        text_tokenized = self.clip_tokenizer(processed_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            text_embedding = self.clip_model.get_text_features(**text_tokenized).cpu().numpy()
            text_embedding = self.normalize(text_embedding)
            _, indices = self.index.search(text_embedding, k)

            indices = indices.flatten()
            print(indices)
            if len(indices) == 0:
                return []

            id_tuple = tuple(int(i) for i in indices)

            return self.get_image_feature_by_tuple(id_tuple)

    def download_csv(self, data):
        def handle_folder_id(id):
            if id < 10:
                return f'0{id}'
            return f'{id}'
        
        def handle_subfolder_id(id):
            if id < 10:
                return f'00{id}'
            return f'0{id}'
        
        df = pd.DataFrame({
            'Folder': [f'L{handle_folder_id(row["folder_id"])}_V{handle_subfolder_id(row["child_folder_id"])}' for row in data],
            'Frame index': [row['frame_mapping_index'] for row in data]
        })

        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return output

    def close(self):
        self.db_cursor.close()
        self.db_connection.close()



