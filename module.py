# model_singleton.py
import os
from pathlib import Path
import tensorflow as tf
from mrcnn import model as modellib
from threading import Lock
from functools import lru_cache
from detect_food import InferenceConfig, get_food_classes
from Plateonlydetect import PlateInferenceConfig

class ModelSingleton:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelSingleton, cls).__new__(cls)
                cls._instance.food_model = None
                cls._instance.plate_model = None
                cls._instance.class_names = None
        return cls._instance
    
    @lru_cache(maxsize=1)
    def load_models(self):
        """Load models once and cache them"""
        ROOT_DIR = Path(__file__).parent.absolute()
        
        # Initialize food model
        food_config = InferenceConfig()
        self.food_model = modellib.MaskRCNN(
            mode="inference", 
            config=food_config, 
            model_dir=str(ROOT_DIR / "logs")
        )
        self.food_model.load_weights(
            str(ROOT_DIR / "logs/food20241024T0144/mask_rcnn_food_0300.h5"), 
            by_name=True
        )
        
        # Load class names
        meta_path = ROOT_DIR / "food-recognition-dataset/meta.json"
        self.class_names = get_food_classes(str(meta_path))
        
        # Initialize plate model
        plate_config = PlateInferenceConfig()
        self.plate_model = modellib.MaskRCNN(
            mode="inference", 
            config=plate_config, 
            model_dir=str(ROOT_DIR / "logs")
        )
        self.plate_model.load_weights(
            str(ROOT_DIR / "logs/mask_rcnn_plate_final.h5"), 
            by_name=True
        )
        
        return self.food_model, self.plate_model, self.class_names

# run_volume_estimation.py
from flask import Flask, jsonify, request
import json
import requests
from pathlib import Path
import tensorflow as tf
from model_singleton import ModelSingleton
from detect_food import detect_food, preprocess_image, filter_similar_detections, convert_to_coco_format
from Plateonlydetect import detect_plates
from threading import Lock
import numpy as np
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)
model_lock = Lock()

# Load configuration
CONFIG_PATH = Path("test_config.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found.")

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

NUTRITION_API_URL = "https://starfish-app-fycwd.ondigitalocean.app/api/nutrition"

def get_models():
    """Thread-safe model getter"""
    singleton = ModelSingleton()
    with model_lock:
        return singleton.load_models()

def process_single_image(image_path):
    """Process a single image with both food and plate detection"""
    food_model, plate_model, class_names = get_models()
    
    try:
        # Process food detection
        image = Image.open(image_path)
        image_array = preprocess_image(image)
        
        food_results = food_model.detect([image_array], verbose=0)[0]
        filtered_food_results = filter_similar_detections(
            food_results,
            iou_threshold=0.25,
            score_threshold=0.15
        )
        
        # Save food detection results
        coco_data = convert_to_coco_format(image_path, filtered_food_results, class_names)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return_dir = Path("return")
        return_dir.mkdir(exist_ok=True)
        
        food_coco_path = return_dir / f"{base_name}.coco.json"
        with open(food_coco_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        # Process plate detection
        plate_results = plate_model.detect([image_array], verbose=0)[0]
        
        # Save plate detection results
        plate_coco_data = {
            "info": {
                "description": "Plate Detection Results",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "images": [{
                "id": 1,
                "width": image_array.shape[1],
                "height": image_array.shape[0],
                "file_name": os.path.basename(image_path)
            }],
            "categories": [{
                "id": 1,
                "name": "plate",
                "supercategory": "object"
            }],
            "annotations": []
        }
        
        for i in range(len(plate_results['rois'])):
            y1, x1, y2, x2 = plate_results['rois'][i]
            plate_coco_data["annotations"].append({
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "area": float((x2-x1) * (y2-y1)),
                "segmentation": [],  # Add proper segmentation if needed
                "iscrowd": 0,
                "score": float(plate_results['scores'][i])
            })
        
        plate_coco_path = return_dir / "annotations.json"
        with open(plate_coco_path, 'w') as f:
            json.dump(plate_coco_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    try:
        # Set up GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # Get image path from request
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({"error": "No image path provided"}), 400
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404
        
        # Process the image
        success = process_single_image(image_path)
        
        if success:
            return jsonify({"message": "Image processed successfully"}), 200
        else:
            return jsonify({"error": "Failed to process image"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# module.py
def run_script():
    """Run the detection pipeline"""
    TEST_IMAGES_DIR = Path("test_images")
    
    if not TEST_IMAGES_DIR.exists():
        TEST_IMAGES_DIR.mkdir(parents=True)
        return False
    
    success = True
    for image_path in TEST_IMAGES_DIR.glob("*.[jp][pn][g]"):
        try:
            process_single_image(str(image_path))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            success = False
    
    return success

if __name__ == '__main__':
    from waitress import serve
    # Configure logging if needed
    serve(app, host='0.0.0.0', port=5000, threads=4)