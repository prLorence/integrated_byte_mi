from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import json
import logging
import requests
from datetime import datetime
from PIL import Image
from model_singleton import ModelSingleton
from run_detection_pipeline import run_detection_pipeline
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.reconstruction.volume_calculator import VolumeCalculator
from src.utils.merge_coco import merge_coco_annotations
from volume_estimation import process_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

NUTRITION_API_URL = "https://starfish-app-fycwd.ondigitalocean.app/api/nutrition"

try:
    with open("test_config.json", 'r') as f:
        config = json.load(f)
    model_singleton = ModelSingleton()
    # Explicitly load models
    model_registry, weights, class_names = model_singleton.load_models()
    logger.info("ModelSingleton and models initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ModelSingleton: {e}")
    raise

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Validate required form fields
        required_fields = ['rgb_image', 'depth_image', 'rgb_meta', 'depth_meta']
        if not all(field in request.form or field in request.files for field in required_fields):
            missing_fields = [field for field in required_fields if field not in request.form and field not in request.files]
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400

        # Parse metadata
        try:
            rgb_meta = json.loads(request.form.get('rgb_meta'))
            depth_meta = json.loads(request.form.get('depth_meta'))
        except json.JSONDecodeError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid metadata format: {str(e)}'
            }), 400
        

        # Get timestamp for frame ID
        timestamp = rgb_meta.get('timestamp')
        frame_id = f"{timestamp}"

        # Set up directory structure
        base_path = Path("test_images")
        return_path = Path("return")
        for path in [base_path, return_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Save RGB image and metadata
        rgb_image = request.files['rgb_image']
        rgb_path = base_path / f"rgb_frame_{frame_id}.png"
        rgb_image.save(str(rgb_path))
        
        rgb_meta_path = base_path / f"rgb_frame_{frame_id}.meta"
        with open(rgb_meta_path, 'w') as f:
            json.dump(rgb_meta, f, indent=2)

        # Save depth data and metadata
        depth_image = request.files['depth_image']
        depth_data = np.frombuffer(depth_image.read(), dtype=np.uint16)
        depth_array = depth_data.reshape((depth_meta['height'], depth_meta['width']))
        
        depth_path = base_path / f"depth_frame_{frame_id}.raw"
        depth_array.tofile(str(depth_path))
        
        depth_meta_path = base_path / f"depth_frame_{frame_id}.meta"
        with open(depth_meta_path, 'w') as f:
            json.dump(depth_meta, f, indent=2)

        run_detection_pipeline(rgb_path, model_singleton, frame_id)
        # Run preprocessing and volume calculation
        processed_frames = process_frames(config, frame_id)

        macro_response = fetch_nutrition_data(processed_frames['volumes'])

        return jsonify({
            'success': True,
            'data': processed_frames,
            'macronutrients': macro_response,
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def fetch_nutrition_data(volumes):
    # Prepare request data for nutrition API
    nutrition_request = {
        "data": [
            {
                "food_name": obj['object_name'],
                "volume": obj['volume_cups']
            }
            for obj in volumes
        ]
    }
    
    try:
        # Make a POST request to the nutrition API
        response = requests.post(NUTRITION_API_URL, json=nutrition_request)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch nutrition data: {e}", exc_info=True)
        return {"error": "Unable to fetch nutrition data"}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)