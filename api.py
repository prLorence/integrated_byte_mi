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

NUTRITION_API_URL = "https://bytemi-fdc-api-etgrn.ondigitalocean.app/v1/calculate-macros"

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

def fetch_nutrition_data(volumes):
    # Prepare request data for nutrition API
    nutrition_requests = []
    
    for vol in volumes:
        min_volume = max(0, vol['volume_cups'] - vol['uncertainty_cups'])
        max_volume = vol['volume_cups'] + vol['uncertainty_cups']
        
        nutrition_requests.append({
            "min": {
                "food_name": vol['object_name'],
                "volume": min_volume
            },
            "expected": {
                "food_name": vol['object_name'],
                "volume": vol['volume_cups']
            },
            "max": {
                "food_name": vol['object_name'],
                "volume": max_volume
            }
        })
    
    try:
        results = []
        for req in nutrition_requests:
            # Get min, expected, and max nutritional values
            min_response = requests.post(NUTRITION_API_URL, json={"data": [req["min"]]}).json()
            exp_response = requests.post(NUTRITION_API_URL, json={"data": [req["expected"]]}).json()
            max_response = requests.post(NUTRITION_API_URL, json={"data": [req["max"]]}).json()
            
            if not all(response.get('data') for response in [min_response, exp_response, max_response]):
                logger.error(f"Invalid response for {req['expected']['food_name']}")
                continue
                
            food_result = {
                "food_name": req["expected"]["food_name"],
                "volume": {
                    "min": round(req["min"]["volume"], 3),
                    "expected": round(req["expected"]["volume"], 3),
                    "max": round(req["max"]["volume"], 3)
                },
                "macros": {
                    "calories": {
                        "min": round(min_response["data"][0]["macros"]["calories"], 1),
                        "expected": round(exp_response["data"][0]["macros"]["calories"], 1),
                        "max": round(max_response["data"][0]["macros"]["calories"], 1)
                    },
                    "protein": {
                        "min": round(min_response["data"][0]["macros"]["protein"], 1),
                        "expected": round(exp_response["data"][0]["macros"]["protein"], 1),
                        "max": round(max_response["data"][0]["macros"]["protein"], 1)
                    },
                    "carbs": {
                        "min": round(min_response["data"][0]["macros"]["carbs"], 1),
                        "expected": round(exp_response["data"][0]["macros"]["carbs"], 1),
                        "max": round(max_response["data"][0]["macros"]["carbs"], 1)
                    },
                    "fat": {
                        "min": round(min_response["data"][0]["macros"]["fat"], 1),
                        "expected": round(exp_response["data"][0]["macros"]["fat"], 1),
                        "max": round(max_response["data"][0]["macros"]["fat"], 1)
                    }
                },
                "serving_info": exp_response["data"][0]["ServingInfo"],
                "uncertainty": {
                    "volume_uncertainty": vol['uncertainty_cups'],
                    "volume_uncertainty_percent": round((vol['uncertainty_cups'] / vol['volume_cups']) * 100, 1)
                }
            }
            results.append(food_result)
            
        totals = {
            "calories": {
                "min": round(sum(r["macros"]["calories"]["min"] for r in results), 1),
                "expected": round(sum(r["macros"]["calories"]["expected"] for r in results), 1),
                "max": round(sum(r["macros"]["calories"]["max"] for r in results), 1)
            },
            "protein": {
                "min": round(sum(r["macros"]["protein"]["min"] for r in results), 1),
                "expected": round(sum(r["macros"]["protein"]["expected"] for r in results), 1),
                "max": round(sum(r["macros"]["protein"]["max"] for r in results), 1)
            },
            "carbs": {
                "min": round(sum(r["macros"]["carbs"]["min"] for r in results), 1),
                "expected": round(sum(r["macros"]["carbs"]["expected"] for r in results), 1),
                "max": round(sum(r["macros"]["carbs"]["max"] for r in results), 1)
            },
            "fat": {
                "min": round(sum(r["macros"]["fat"]["min"] for r in results), 1),
                "expected": round(sum(r["macros"]["fat"]["expected"] for r in results), 1),
                "max": round(sum(r["macros"]["fat"]["max"] for r in results), 1)
            }
        }
        
        for macro in totals:
            expected = totals[macro]["expected"]
            if expected > 0:
                totals[macro]["uncertainty_percent"] = round(
                    ((totals[macro]["max"] - totals[macro]["min"]) / (2 * expected)) * 100, 1
                )
            else:
                totals[macro]["uncertainty_percent"] = 0
        
        return {
            "data": results,
            "totals": totals,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "number_of_items": len(results)
            }
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch nutrition data: {e}", exc_info=True)
        return {"error": "Unable to fetch nutrition data"}

@app.route('/process', methods=['POST'])
def process_image():
    """Process uploaded image and return volume and nutrition estimates with uncertainties."""
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

        # Run detection pipeline
        run_detection_pipeline(rgb_path, model_singleton, frame_id)
        
        # Run preprocessing and volume calculation
        processed_frames = process_frames(config, frame_id)

        # Get nutrition data with uncertainty ranges
        macro_response = fetch_nutrition_data(processed_frames['volumes'])

        return jsonify({
            'success': True,
            'data': processed_frames,
            'macronutrients': macro_response,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
