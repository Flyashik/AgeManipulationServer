import os
import base64
import subprocess
import shutil
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)
IMAGE_DIR = 'images'
RESULT_DIR = 'result/inference_result'

mutex = threading.Lock()

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if 'imageData' not in data or 'targetAge' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    image_data = data['imageData']
    target_age = data['targetAge']
    
    try:
        target_age = int(target_age)
    except ValueError:
        return jsonify({'error': 'Invalid targetAge'}), 400
    
    image_bytes = base64.b64decode(image_data)
    
    with mutex:
        clear_directory(IMAGE_DIR)
        clear_directory(RESULT_DIR)
        
        image_path = os.path.join(IMAGE_DIR, 'input_image.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        command = [
            'python', 'scripts/inference.py',
            '--exp_dir=result',
            '--checkpoint_path=pretrained_models/sam_ffhq_aging.pt',
            '--data_path=images',
            '--target_age=' + str(target_age)
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': 'Generate script failed: ' + str(e)}), 500
        
        generated_image_path = os.path.join(RESULT_DIR, str(target_age), 'input_image.jpg')
        
        if not os.path.exists(generated_image_path):
            return jsonify({'error': 'Generated image not found'}), 500
        
        with open(generated_image_path, 'rb') as f:
            generated_image_bytes = f.read()
    
    generated_image_base64 = base64.b64encode(generated_image_bytes).decode('utf-8')
    return jsonify({'base64': generated_image_base64})

if __name__ == '__main__':
    app.run(host='192.168.0.11', port=5000)