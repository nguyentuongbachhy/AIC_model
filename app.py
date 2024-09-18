import torch, os, logging, json
from urllib.parse import quote as url_quote
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from utils.ImageTextSearchEngine import ImageTextSearchEngine
from utils.Translation import Translation, TextPreprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load environment variables from .env file
load_dotenv()

# Load database config
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Load file bin
bin_file = 'D:/demo-ai-challenge/model/faiss_normal_ViT.bin'

image_text_search_engine = ImageTextSearchEngine(
    db_config=db_config,
    bin_file=bin_file,
    translator=Translation(),
    text_preprocessing=TextPreprocessing(),
    clip_backbone='ViT-B/32',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Flask app setup
app = Flask(__name__)
CORS(app=app, resources={
    r"/*": {
        "origins": "http://localhost:3000"
    }
})


# Load the image paths from a JSON file
# with open('./assets/image_paths.json') as json_file:
#     json_dict = json.load(json_file)

# @app.route('/insert-all-images/', methods=['GET'])
# def insert_all_images():
#     for _, image_path in json_dict.items():
#         try:
#             image_text_search_engine.insert_image(image_path)
#         except Exception as e:
#             logging.error(f"Error when inserting image: {str(e)}")
#             return jsonify({'Error': str(e)}), 500
#     return jsonify({"Inserted all images successfully"}), 200

@app.route('/image-search', methods=['GET'])
def image_search():
    img_id = int(request.args.get('imgId')) - 1
    k = int(request.args.get('k'))
    try:
        image_paths = image_text_search_engine.search_images_by_id(img_id, k=k)
        return jsonify({'results': image_paths}), 200
    except Exception as e:
        logging.error(f'Error in image-search: {str(e)}')
        return jsonify({'Error': str(e)}), 500

@app.route('/text-search', methods=['GET'])
def text_search():
    text = request.args.get('query')
    k = int(request.args.get('k'))
    print(text)
    try:
        image_paths = image_text_search_engine.search_images_by_text(text, k=k)
        return jsonify({'results': image_paths}), 200
    except Exception as e:
        logging.error(f'Error in text-search: {str(e)}')
        return jsonify({'Error': str(e)}), 500

@app.route('/similar-part-search', methods=['GET'])
def similar_parts_search():
    imgId = int(request.args.get('imgId')) - 1
    x1 = float(request.args.get('x1'))
    x2 = float(request.args.get('x2'))
    y1 = float(request.args.get('y1'))
    y2 = float(request.args.get('y2'))
    k = int(request.args.get('k'))
    try:
        image_paths = image_text_search_engine.search_similar_images_for_part(imgId=imgId, x1=x1, x2=x2, y1=y1, y2=y2, k=k)
        image_paths = [{"id": int(result["id"]), "image_path": result["image_path"]} for result in image_paths]
        return jsonify({'results': image_paths}), 200
    except Exception as e:
        logging.error(f'Error in similar-part-search: {str(e)}')
        return jsonify({'Error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
