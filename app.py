import torch, os, logging
from urllib.parse import quote as url_quote
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from flask_cors import CORS
from utils.ImageTextSearchEngine import ImageTextSearchEngine
from utils.Translation import Translation
from utils.TextProcessor import TextProcessor
from io import BytesIO

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
bin_file = 'D:/AIC/model/faiss_normal_ViT.bin'

image_text_search_engine = ImageTextSearchEngine(
    db_config=db_config,
    bin_file=bin_file,
    translator=Translation(),
    text_preprocessing=TextProcessor(),
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

@app.route('/image-search', methods=['GET'])
def image_search():
    img_id = int(request.args.get('imgId'))
    k = int(request.args.get('k'))
    print(img_id)
    print(k)
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


@app.route('/download-csv', methods=['POST'])
def export_csv():
    data = request.json
    output = image_text_search_engine.download_csv(data=data)
    return send_file(output, as_attachment=True, download_name='output.csv', mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
