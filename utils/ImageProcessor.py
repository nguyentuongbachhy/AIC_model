import os, json, torch, requests, concurrent.futures, time
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from io import BytesIO


class ImageProcessor:
    def __init__(self, device, max_threads=6, max_retries=3, retry_delay=5):
        self.device = device
        self.max_threads = max_threads
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")



        # Initialize Dropbox
        # self.dbx = dropbox.Dropbox(
        #     oauth2_refresh_token=dropbox_config['refresh_token'],
        #     app_key=dropbox_config['app_key'],
        #     app_secret=dropbox_config['app_secret']
        # )
        # # Verify Dropbox connection
        # try:
        #     self.dbx.users_get_current_account()
        # except AuthError:
        #     raise ValueError("Invalid Dropbox authentication credentials.")

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

    def open_image(self, image_url):
        attempt = 0  # Counter for retry attempts
        while attempt < self.max_retries:
            try:
                # Fetch the image from the URL
                response = requests.get(image_url)
                response.raise_for_status()

                # Use BytesIO to open the image
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                image = image.convert('RGB')  # Convert to RGB to handle different formats
                return image

            except requests.exceptions.RequestException as e:
                print(f"Error fetching the image from {image_url} (attempt {attempt + 1}/{self.max_retries}): {e}")
            except (IOError, OSError) as e:
                print(f"Error opening the image from {image_url} (attempt {attempt + 1}/{self.max_retries}): {e}")

            # Increment the attempt counter and wait before retrying
            attempt += 1
            if attempt < self.max_retries:
                print(f"Waiting {self.retry_delay} seconds before retrying...")
                time.sleep(self.retry_delay)

        # Return None if all attempts fail
        print(f"Failed to open image from {image_url} after {self.max_retries} attempts.")
        return None

    def extract_clip_features(self, image):
        inputs = self.clip_preprocess(images=image, return_tensors="pt").to(device)

        # Extract image features only
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features.cpu().numpy().flatten()

    # def upload_to_dropbox(self, image, image_name, folder='aic', fallback_folder='aic_backup', retry_count=3):
    #     buffer = BytesIO()
    #     image.save(buffer, format='JPEG')
    #     buffer.seek(0)

    #     dropbox_path = f'/{folder}/{image_name}.jpg'
    #     fallback_path = f'/{fallback_folder}/{image_name}.jpg' if fallback_folder else dropbox_path

    #     for attempt in range(retry_count):
    #         try:
    #             try:
    #                 # Check for shared links in both main folder and fallback folder
    #                 shared_links = self.dbx.sharing_list_shared_links(path=dropbox_path).links
    #                 if not shared_links:
    #                     shared_links = self.dbx.sharing_list_shared_links(path=fallback_path).links

    #                 if shared_links:
    #                     return shared_links[0].url.replace("&dl=0", "&raw=1")

    #             except dropbox.exceptions.ApiError as e:
    #                 if isinstance(e.error, dropbox.sharing.ListSharedLinksError) and e.error.is_path() and e.error.get_path().is_not_found():
    #                     # The path is not found, proceed to upload the file
    #                     pass
    #                 else:
    #                     # Reraise any other API errors encountered while listing shared links
    #                     raise

    #             # If no shared links exist, upload to the main path
    #             self.dbx.files_upload(buffer.read(), dropbox_path, mode=WriteMode.overwrite)

    #             # Create a new shared link
    #             shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(dropbox_path)
    #             return shared_link_metadata.url.replace("&dl=0", "&raw=1")

    #         except dropbox.exceptions.ApiError as e:
    #             error = e.error

    #             # Check if the shared link already exists
    #             if isinstance(error, dropbox.sharing.CreateSharedLinkWithSettingsError) and error.is_shared_link_already_exists():
    #                 # Extract and return the existing shared link
    #                 shared_link_metadata = error.get_shared_link_already_exists().metadata
    #                 return shared_link_metadata.url.replace("&dl=0", "&raw=1")

    #             # Handle other API errors
    #             print(f"Error during upload to Dropbox: {e}")
    #             if attempt < retry_count - 1:
    #                 time.sleep(2 ** attempt)  # Exponential backoff
    #             else:
    #                 # Retry upload to fallback folder if an error occurs
    #                 print(f"Retrying upload to fallback folder '{fallback_folder}'")
    #                 try:
    #                     self.dbx.files_upload(buffer.read(), fallback_path, mode=WriteMode.overwrite)
    #                     # Create a new shared link for the fallback folder
    #                     shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(fallback_path)
    #                     return shared_link_metadata.url.replace("&dl=0", "&raw=1")
    #                 except dropbox.exceptions.ApiError as fallback_e:
    #                     print(f"Error during upload to fallback folder: {fallback_e}")
    #                     raise

    def process_images_in_jsonfile(self, filename, filepath):
        
        child_folder_data = {}

        def process_image(url, index, frame_index):
            image = self.open_image(url)
            vector_feature = self.extract_clip_features(image)
            
            print(f'Extracted image features from {filename}_{index} successfully!')

            if url:
                child_folder_data[int(index)] = {
                    "url": url,
                    "vector_feature": vector_feature.tolist(),
                    "frame_index": int(frame_index),
                }

        data = self.read_json(filepath)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_image, entry['url'], key, entry['frame_index']) for key, entry in data.items()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing image: {e}")

        self.save_json(filepath=f'D:/AIC/model/assets/results/{filename}_details.json', data=child_folder_data)

    def process_images_in_folder(self, root_folder):
        
        for file in os.listdir(root_folder):
            filename = file.replace("_details.json", "")
            filepath = f'{root_folder}/{file}'
            self.process_images_in_jsonfile(filename, filepath)
        
        print('Successfully processed all folders')




# Constants for paths and configuration
DROPBOX_JSON = 'D:/AIC/model/assets/credentials.json'
ROOT_FOLDER = 'D:/AIC/model/assets/storage'

if __name__ == '__main__':
    with open(DROPBOX_JSON) as file:
        dropbox_conf = json.load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imageProcessor = ImageProcessor(device=device)
    imageProcessor.process_images_in_folder(ROOT_FOLDER)