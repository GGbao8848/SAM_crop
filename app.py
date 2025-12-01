import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import SAM
from PIL import Image
import io

app = Flask(__name__)

# Configuration
MODELS_FOLDER = 'models'
os.makedirs(MODELS_FOLDER, exist_ok=True)
app.config['MODELS_FOLDER'] = MODELS_FOLDER

# Global model variable
model = None
current_model_name = None

def load_model(model_name):
    global model, current_model_name
    try:
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
        if os.path.exists(model_path):
            model = SAM(model_path)
            current_model_name = model_name
            print(f"Loaded model: {model_name}")
            return True
        return False
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return False

# Initial load (try to find any .pt file)
available_models = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pt')]
if available_models:
    load_model(available_models[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    models = [f for f in os.listdir(app.config['MODELS_FOLDER']) if f.endswith('.pt')]
    return jsonify({'models': models, 'current': current_model_name})

@app.route('/api/set_model', methods=['POST'])
def set_model():
    data = request.json
    model_name = data.get('model')
    if not model_name:
        return jsonify({'error': 'Model name required'}), 400
    
    if load_model(model_name):
        return jsonify({'status': 'success', 'current': current_model_name})
    else:
        return jsonify({'error': 'Failed to load model'}), 500

@app.route('/api/list_images', methods=['POST'])
def list_images():
    data = request.json
    path = data.get('path')
    if not path or not os.path.isdir(path):
        return jsonify({'error': 'Invalid directory path'}), 400
    
    images = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    try:
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(valid_exts):
                images.append(f)
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image')
def get_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path)

@app.route('/api/segment', methods=['POST'])
def segment():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    image_path = data.get('image_path')
    points = data.get('points', [])
    labels = data.get('labels', [])
    box = data.get('box', None) # [x1, y1, x2, y2]

    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Image path required and must exist'}), 400

    try:
        # Construct arguments
        kwargs = {'source': image_path}
        
        if points:
            kwargs['points'] = [points]
            kwargs['labels'] = [labels]
        
        if box:
            kwargs['bboxes'] = [box]

        results = model(**kwargs)
        
        if not results:
            return jsonify({'error': 'No results'}), 500

        mask_tensor = results[0].masks.data[0]
        mask_np = mask_tensor.cpu().numpy().astype('uint8') * 255
        
        orig_h, orig_w = results[0].orig_shape
        if mask_np.shape != (orig_h, orig_w):
            mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        _, buffer = cv2.imencode('.png', mask_np)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'mask': f'data:image/png;base64,{mask_b64}'})

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop', methods=['POST'])
def crop_image():
    data = request.json
    image_path = data.get('image_path')
    mask_data = data.get('mask')

    if not image_path or not mask_data:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        if ',' in mask_data:
            mask_data = mask_data.split(',')[1]
        mask_bytes = base64.b64decode(mask_data)
        mask_arr = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
             return jsonify({'error': 'Empty mask'}), 400
             
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        img_crop = img[y_min:y_max+1, x_min:x_max+1]
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        
        b, g, r = cv2.split(img_crop)
        rgba = [b, g, r, mask_crop]
        dst = cv2.merge(rgba, 4)
        
        pil_img = Image.fromarray(dst)
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        return jsonify({'cropped_image': f'data:image/png;base64,{img_str}'})

    except Exception as e:
        print(f"Crop error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_crop', methods=['POST'])
def batch_crop():
    data = request.json
    items = data.get('items', []) # List of {image_path, mask}
    
    if not items:
        return jsonify({'error': 'No items to process'}), 400
        
    results = []
    errors = []
    
    for item in items:
        image_path = item.get('image_path')
        mask_data = item.get('mask')
        
        if not image_path or not mask_data:
            continue
            
        try:
            # Create crops directory
            source_dir = os.path.dirname(image_path)
            crops_dir = os.path.join(source_dir, 'crops')
            os.makedirs(crops_dir, exist_ok=True)
            
            # Decode mask
            if ',' in mask_data:
                mask_data = mask_data.split(',')[1]
            mask_bytes = base64.b64decode(mask_data)
            mask_arr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            
            # Read image
            img = cv2.imread(image_path)
            # Keep BGR for saving with cv2
            
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0:
                 errors.append(f"Empty mask for {os.path.basename(image_path)}")
                 continue
                 
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            img_crop = img[y_min:y_max+1, x_min:x_max+1]
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Add alpha channel
            b, g, r = cv2.split(img_crop)
            rgba = [b, g, r, mask_crop]
            dst = cv2.merge(rgba, 4)
            
            # Save to disk
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(crops_dir, f"{name}_crop.png")
            
            cv2.imwrite(save_path, dst)
            results.append(save_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            errors.append(str(e))
            
    return jsonify({'saved': results, 'errors': errors})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
