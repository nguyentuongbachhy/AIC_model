import json, faiss, mysql.connector, clip, torch, requests, os, numpy as np
from dotenv import load_dotenv

load_dotenv()

# Global variables
FOLDER_PATH = 'D:/demo-ai-challenge/model/assets/results'

# Load mapping file and sql config
with open('D:/demo-ai-challenge/model/assets/objects.basic.json', 'r') as file:
    object_mapping = json.load(file)

color_mapping = {
    "white": 1,
    "red": 2,
    "yellow": 3,
    "orange": 4,
    "green": 5,
    "cyan": 6,
    "blue": 7,
    "purple": 8,
    "black": 9
}

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
index = faiss.IndexFlatL2(512)

for filename in os.listdir(FOLDER_PATH):
    arr_id = filename.split('_')
    folder_id = int(arr_id[0].replace('L0', '').replace('L', ''))
    child_folder_id = int(arr_id[1].replace('V0', '').replace('V', ''))

    file_path = os.path.join(FOLDER_PATH, filename)
    with open(file_path, 'r') as file:
        listData = json.load(file)
    
    insert_img_feature_sql = "INSERT INTO image_features (folder_id, child_folder_id, id_frame, image_path, frame_mapping_index) VALUES (%s, %s, %s, %s, %s)"
    insert_img_object_sql = "INSERT INTO image_objects (image_id, object_id, object_count) VALUES (%s, %s, %s)"
    insert_img_color_sql = "INSERT INTO image_colors (image_id, color_id) VALUES (%s, %s)"

    for key, entry in listData.items():
        db_cursor.execute(insert_img_feature_sql, (folder_id, child_folder_id, key, entry['url'], entry['frame_index']))
        image_id = db_cursor.lastrowid

        for object_name, object_count in entry['yolo'].items():
            object_id = object_mapping[object_name]
            db_cursor.execute(insert_img_object_sql, (image_id, object_id, object_count))
        
        filtered_colors = set(entry['main_color'])

        for color in filtered_colors:
            color_id = color_mapping[color]
            db_cursor.execute(insert_img_color_sql, (image_id, color_id))

        vector_feature = np.array(entry['vector_feature'], dtype=np.float32)
        
        # Kiểm tra kích thước vector để đảm bảo nó có 512 chiều
        if vector_feature.shape[0] != 512:
            print(f"Error: Vector feature in {filename} (key: {key}) has incorrect dimension: {vector_feature.shape}")
            continue
        
        # Reshape để phù hợp với FAISS (1, 512)
        vector_feature = vector_feature.reshape(1, -1)

        # Thêm vào FAISS index
        index.add(vector_feature)

    print(f'Inseted all entries in {filename} successfully')

db_connection.commit()

faiss.write_index(index, "D:/demo-ai-challenge/model/faiss_normal_ViT.bin")

db_cursor.close()
db_connection.close()





