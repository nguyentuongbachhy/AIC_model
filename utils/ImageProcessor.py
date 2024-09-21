import os, json, torch, clip, time, dropbox, pandas as pd, concurrent.futures
from PIL import Image
from io import BytesIO
from dropbox.exceptions import AuthError
from dropbox.files import WriteMode



class ImageProcessor:
    def __init__(self, dropbox_config, device, max_threads=6, clip_model='ViT-B/32'):
        self.device = device
        self.max_threads = max_threads
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)

        # Initialize Dropbox
        self.dbx = dropbox.Dropbox(
            oauth2_refresh_token=dropbox_config['refresh_token'],
            app_key=dropbox_config['app_key'],
            app_secret=dropbox_config['app_secret']
        )
        # Verify Dropbox connection
        try:
            self.dbx.users_get_current_account()
        except AuthError:
            raise ValueError("Invalid Dropbox authentication credentials.")

    def read_json(self, filepath):
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f'File {filepath} does not exist.')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {filepath}: {e}")
        return None
        
    def save_json(self, filepath, data):
        try:
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")

    def extract_clip_features(self, image):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        return image_features.cpu().numpy().flatten()

    def upload_to_dropbox(self, image, image_name, folder='aic', fallback_folder='aic_backup', retry_count=3):
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)

        dropbox_path = f'/{folder}/{image_name}.jpg'
        fallback_path = f'/{fallback_folder}/{image_name}.jpg' if fallback_folder else dropbox_path

        for attempt in range(retry_count):
            try:
                try:
                    # Check for shared links in both main folder and fallback folder
                    shared_links = self.dbx.sharing_list_shared_links(path=dropbox_path).links
                    if not shared_links:
                        shared_links = self.dbx.sharing_list_shared_links(path=fallback_path).links

                    if shared_links:
                        return shared_links[0].url.replace("&dl=0", "&raw=1")

                except dropbox.exceptions.ApiError as e:
                    if isinstance(e.error, dropbox.sharing.ListSharedLinksError) and e.error.is_path() and e.error.get_path().is_not_found():
                        # The path is not found, proceed to upload the file
                        pass
                    else:
                        # Reraise any other API errors encountered while listing shared links
                        raise

                # If no shared links exist, upload to the main path
                self.dbx.files_upload(buffer.read(), dropbox_path, mode=WriteMode.overwrite)

                # Create a new shared link
                shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(dropbox_path)
                return shared_link_metadata.url.replace("&dl=0", "&raw=1")

            except dropbox.exceptions.ApiError as e:
                error = e.error

                # Check if the shared link already exists
                if isinstance(error, dropbox.sharing.CreateSharedLinkWithSettingsError) and error.is_shared_link_already_exists():
                    # Extract and return the existing shared link
                    shared_link_metadata = error.get_shared_link_already_exists().metadata
                    return shared_link_metadata.url.replace("&dl=0", "&raw=1")

                # Handle other API errors
                print(f"Error during upload to Dropbox: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Retry upload to fallback folder if an error occurs
                    print(f"Retrying upload to fallback folder '{fallback_folder}'")
                    try:
                        self.dbx.files_upload(buffer.read(), fallback_path, mode=WriteMode.overwrite)
                        # Create a new shared link for the fallback folder
                        shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(fallback_path)
                        return shared_link_metadata.url.replace("&dl=0", "&raw=1")
                    except dropbox.exceptions.ApiError as fallback_e:
                        print(f"Error during upload to fallback folder: {fallback_e}")
                        raise

    def process_images_in_subfolder(self, subfolder_name, folder_path):
        keyframes_path = f'{folder_path}/keyframes/{subfolder_name}'
        mapping_path = f'{folder_path}/mapping/{subfolder_name}.csv'

        child_folder_data = {}

        def handleIndex(index):
            if index < 10: return f'00{index}'
            elif index < 100: return f'0{index}'
            return f'{index}'

        def process_image(index, frame_index):
            fixed_index = handleIndex(index)
            print(f'Processing {subfolder_name}_{fixed_index}')
            image_path = os.path.join(keyframes_path, f'{fixed_index}.jpg')
            image = Image.open(image_path)

            vector_feature = self.extract_clip_features(image)
            url = self.upload_to_dropbox(image, f'{subfolder_name}_{fixed_index}')
            
            if url:
                child_folder_data[int(index)] = {
                    "url": url,
                    "vector_feature": vector_feature.tolist(),
                    "frame_index": int(frame_index),
                }

        df = pd.read_csv(mapping_path, usecols=['n', 'frame_idx'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_image, row['n'], row['frame_idx']) for _, row in df.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing image: {e}")

        self.save_json(filepath=f'D:/AIC/model/assets/results/{subfolder_name}_details.json', data=child_folder_data)

    def process_images_in_folder(self, root_folder):
        
        folder_path = f'{root_folder}/keyframes'
        for filename in os.listdir(folder_path):
            self.process_images_in_subfolder(filename, root_folder)
        
        print('Successfully processed all folders')




# Constants for paths and configuration
DROPBOX_JSON = 'D:/AIC/model/assets/credentials.json'
ROOT_FOLDER = 'D:/AIC/model/assets'

if __name__ == '__main__':
    with open(DROPBOX_JSON) as file:
        dropbox_conf = json.load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imageProcessor = ImageProcessor(dropbox_config=dropbox_conf, device=device)
    imageProcessor.process_images_in_folder(ROOT_FOLDER)