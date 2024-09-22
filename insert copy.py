import json, faiss, mysql.connector, clip, torch, requests, os, numpy as np
from dotenv import load_dotenv

load_dotenv()

# Global variables
FOLDER_PATH = 'D:/AIC/model/assets/results'

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}


# Load model
device = 'cuda' if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# # Connect to MySQL
db_connection = mysql.connector.connect(**db_config)
db_cursor = db_connection.cursor()


# Initial Faiss index
index = faiss.IndexIDMap(faiss.IndexFlatL2(512))

for filename in os.listdir(FOLDER_PATH):
    arr_id = filename.split('_')
    folder_id = int(arr_id[0].replace('L0', '').replace('L', ''))
    child_folder_id = int(arr_id[1].replace('V0', '').replace('V', ''))

    file_path = os.path.join(FOLDER_PATH, filename)
    with open(file_path, 'r') as file:
        listData = json.load(file)
    
    insert_img_feature_sql = "INSERT INTO image_features (folder_id, child_folder_id, id_frame, image_path, frame_mapping_index, vector_features) VALUES (%s, %s, %s, %s, %s, %s)"

    for key, entry in listData.items():
        vector_features = np.array(entry['vector_feature'], dtype=np.float32)
        if vector_features.shape[0] != 512:
            print(f"Error: Vector feature in {filename} (key: {key}) has incorrect dimension: {vector_features.shape}")
            continue
        
        vector_features = vector_features.reshape(1, -1)
        vector_blob = vector_features.tobytes()
        
        db_cursor.execute(insert_img_feature_sql, (folder_id, child_folder_id, key, entry['url'], entry['frame_index'], vector_blob))
        image_id = db_cursor.lastrowid

        index.add_with_ids(vector_features, np.array([image_id], dtype=np.int64))

    print(f'Inserted all entries in {filename} successfully')

db_connection.commit()

faiss.write_index(index, "D:/AIC/model/faiss_normal_ViT.bin")

db_cursor.close()
db_connection.close()






