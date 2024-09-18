import faiss
import clip
import torch
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

json_file_path = 'D:/demo-ai-challenge/model/assets/keyframes'


def process_image(file_path):
    image = Image.open(file_path)
    with torch.no_grad():
        image_features = preprocess(image).unsqueeze(0).to(device)
        image_vector = model.encode_image(image_features).cpu().numpy()
        faiss.normalize_L2(image_vector)
    image.close()  # Close image after processing to free up memory
    return image_vector


def process_folder(subfolder_path, index):
    keyframes_path = os.path.join(subfolder_path, 'keyframes')
    image_vectors = []

    # Collect all image paths
    image_paths = [os.path.join(keyframes_path, filename) for filename in os.listdir(keyframes_path)]

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        image_vectors = list(executor.map(process_image, image_paths))

    # Add all image vectors to the FAISS index
    for vector in image_vectors:
        index.add(vector)


def main():
    for folder in os.listdir(json_file_path):
        number = folder.replace('L0', '').replace('L', '')
        folder_path = os.path.join(json_file_path, folder)
        index = faiss.IndexFlatL2(512)
        
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            process_folder(subfolder_path, index)

        faiss.write_index(index, f"faiss_normal_{number}.bin")
        print(f'Write in faiss_normal_{number}.bin successfully')


if __name__ == "__main__":
    main()
