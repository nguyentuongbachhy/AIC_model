import dropbox.exceptions
import os, json, torch, clip, time, math, dropbox
from PIL import Image
from ultralytics import YOLO
from io import BytesIO
from colorthief import ColorThief
from collections import Counter

class ImageProcessor:
    def __init__(self, dropbox_config, basic_colors, device, clip_model='ViT-B/32', yolo_model='D:/demo-ai-challenge/model/yolov8m.pt'):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.yolo_model = YOLO(yolo_model)
        self.basic_colors = basic_colors
        self.access_token = dropbox_config['access_token']
        
        self.dbx = dropbox.Dropbox(self.access_token)


    def read_json(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    
    def save_json(self, file_path, data):
        with open(file_path, 'w') as file:
            json.dump(file, data, indent=4)
    
    def extract_clip_features(self, image):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        
        return image_features.cpu().numpy().flatten()
    
    def detect_objects(self, image_path):
        results = self.yolo_model.predict(image_path,  conf=0.25)
        object_counts = {}
        
        for result in results:
            for box in result.boxes:
                label = result.names[box.cls[0].item()]
                object_counts[label] = object_counts.get(label, 0) + 1
        
        return object_counts
    

    def upload_to_dropbox(self, image, folder, image_name, retry_count=3):
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        dropbox_path = f'/{folder}/{image_name}.jpg'

        for attempt in range(retry_count):
            try:
                self.dbx.files_upload(buffer.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(dropbox_path)
                return shared_link_metadata.url.replace("?dl=0", "?raw=1")

            except dropbox.exceptions.ApiError as e:
                print(f'Error during upload to Dropbox: {e}')
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def get_main_colors(self, image_path, threshold=0.3):
        image = Image.open(image_path)
        image = image.convert('RGB')

        pixels = list(image.getdata())

        pixel_count = Counter(pixels)
        total_pixels = len(pixel_count)

        color_percentages = {
            color: count / total_pixels for color, count in pixel_count.items()
        }

        dominant_colors = {
            color: percentage for color, percentage in color_percentages.items() if percentage > threshold
        }

        dominant_colors_names = [self.get_closest_color(color) for color in dominant_colors]

        return dominant_colors_names
    
    def get_closest_color(self, rgb_color):
        closest_color = None
        min_distance = float('inf')
        for color_name, color_value in self.basic_colors.items():
            distance = math.sqrt(
                (rgb_color[0] - color_value[0]) ** 2 +
                (rgb_color[1] - color_value[1]) ** 2 +
                (rgb_color[2] - color_value[2]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    def process_images_in_subfolder(self, base_folder):
        for subfolder in os.listdir(base_folder):
            subfolder_path = os.path.join(base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                mapping_path = os.path.join(subfolder_path, f'{subfolder}.json')
                mapping_file = self.read_json(mapping_path)

                subfolder_data = {}
                keyframes_folder = os.path.join(subfolder_path, 'keyframes')

                for index, frame_index in mapping_file.items():
                    # if int(index) > 47:
                    #     break
                    image_path = os.path.join(keyframes_folder, f'{index}.jpg')
                    image = Image.open(image_path)

                    vector_feature = self.extract_clip_features(image)

                    label_counts = self.detect_objects(image_path)

                    main_colors = self.get_main_colors(image_path)
                    

                    url = self.upload_to_dropbox(image, 'aic_storage', f'{subfolder}_{index}')

                    subfolder_data[index] = {
                        "url": url,
                        "vector_feature": vector_feature.tolist(),
                        "frame_index": frame_index,
                        "yolo": label_counts,
                        "main_color": main_colors
                    }

                with open(f'D:/demo-ai-challenge/model/assets/results/{subfolder}_details.json', 'w') as file:
                    json.dump(subfolder_data, file, indent=4)

    def process_images_in_folder(self, root_folder):
        for foldername in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, foldername)
            self.process_images_in_subfolder(folder_path)
            print(f'Processed all subfolder in {foldername} successfully')


DROPBOX_JSON = 'D:/demo-ai-challenge/model/assets/credentials.json'
ROOT_FOLDER = 'D:/demo-ai-challenge/model/assets/keyframes/'
OUTPUT_JSON = 'D:/demo-ai-challenge/model/assets/result.json'

basic_colors = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "green": (0, 128, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "black": (0, 0, 0)
}

if __name__ == '__main__':
    with open(DROPBOX_JSON) as file:
        dropbox_conf = json.load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imageProcessor = ImageProcessor(dropbox_conf, basic_colors=basic_colors, device=device)
    imageProcessor.process_images_in_folder(ROOT_FOLDER)