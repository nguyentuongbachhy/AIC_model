# import cloudinary
# import cloudinary.api
# import json
# import os

# import cloudinary.exceptions
# import cloudinary.uploader

# CLOUDINARY_JSON = 'D:/demo-ai-challenge/model/assets/credentials.json'

# IMAGE_PATHS_JSON = 'D:/demo-ai-challenge/model/assets/image_paths.json'

# FOLDER_PATH = 'D:/demo-ai-challenge/model/assets/images'

# class MyCloudinary:
#     def __init__(self, cloud_conf):
#         self.cloud_name = cloud_conf['cloud_name']
#         self.api_key = cloud_conf['api_key']
#         self.api_secret = cloud_conf['api_secret']

#         # Configure cloudinary
#         cloudinary.config(
#             cloud_name=self.cloud_name,
#             api_key=self.api_key,
#             api_secret = self.api_secret
#         )

#     def upload_image(self, image_path: str, folder: str = None):
#         try:
#             response = cloudinary.uploader.upload(image_path, folder=folder)
#             return response['secure_url']
#         except cloudinary.exceptions.Error as e:
#             print(f"An error when uploading image: {e}")
#             return None

#     def upload_folder(self, source_folder:str, target_folder:str = None):
#         if os.path.exists(source_folder):
#             urls = []
#             for filename in os.listdir(source_folder):
#                 file_path = os.path.join(source_folder, filename)
#                 url = self.upload_image(file_path, folder=target_folder)
#                 urls.append(url)

#             with open(IMAGE_PATHS_JSON, 'w') as image_paths_file:
#                 json.dump(urls, image_paths_file, indent=4)
#             return "All images are uploaded to cloudinary"
#         else:
#             print("Folder does not exist!")
#             return None

#     def delete_all_images(self, resource_type: str = 'image', batch_size: int = 100):
#         next_cursor = None

#         while True:
#             try:
#                 # Fetch resources to delete
#                 response = cloudinary.api.resources(
#                     type='upload',
#                     resource_type=resource_type,
#                     max_results=batch_size,  # Fetch the maximum number of results
#                     next_cursor=next_cursor
#                 )

#                 public_ids = [resource['public_id'] for resource in response.get('resources', [])]

#                 if public_ids:
#                     # Delete the resources by their public IDs
#                     delete_response = cloudinary.api.delete_resources(public_ids)

#                     print(f"Deleted: {delete_response['deleted']}")
                
#                 # Check if there is a next page of resources
#                 next_cursor = response.get('next_cursor')
                
#                 if not next_cursor:
#                     break

#             except cloudinary.exceptions.Error as e:
#                 print(f"An error occurred while deleting: {e}")
#                 break

#     def get_all_urls(self, resource_type: str = 'image', max_results: int = 500):
#         urls = []
#         next_cursor = None

#         while True:
#             try:
#                 # Fetch resources from Cloudinary
#                 response = cloudinary.api.resources(
#                     type='upload',
#                     resource_type=resource_type,
#                     max_results=max_results,
#                     next_cursor=next_cursor
#                 )

#                 # Iterate through the resources and collect URLs
#                 for resource in response.get('resources', []):
#                     urls.append(resource.get('url'))

#                 # Check if there is a next page of resources
#                 next_cursor = response.get('next_cursor')
                
#                 if not next_cursor:
#                     break

#             except cloudinary.exceptions.Error as e:
#                 print(f"An error occurred: {e}")
#                 break
        
#         return urls
    

# if __name__ == '__main__':
    # with open(CLOUDINARY_JSON) as file:
    #     cloud_conf = json.load(file)

    # my_cloudinary = MyCloudinary(cloud_conf=cloud_conf)

    # my_cloudinary.delete_all_images()