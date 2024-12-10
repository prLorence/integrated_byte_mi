import os
import sys
import json
import numpy as np
import tensorflow as tf
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Food Detection Config Classes
class FoodConfig(Config):
    NAME = "food"
    
    meta_path = os.path.join(os.path.abspath("."), "food-recognition-dataset", "meta.json")
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
        NUM_CLASSES = 1 + len(meta_data['classes'])
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class FoodInferenceConfig(FoodConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.15
    DETECTION_MAX_INSTANCES = 25
    DETECTION_NMS_THRESHOLD = 0.25

# Plate Detection Config Class
class PlateInferenceConfig(Config):
    NAME = "plate"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + plate
    DETECTION_MIN_CONFIDENCE = 0.5
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

def load_models(food_weights_path, plate_weights_path):
    """Load both food and plate detection models"""
    # Initialize configs
    food_config = FoodInferenceConfig()
    plate_config = PlateInferenceConfig()
    
    # Initialize models
    food_model = modellib.MaskRCNN(mode="inference", config=food_config, model_dir="logs")
    plate_model = modellib.MaskRCNN(mode="inference", config=plate_config, model_dir="logs")
    
    # Load weights
    food_model.load_weights(food_weights_path, by_name=True)
    plate_model.load_weights(plate_weights_path, by_name=True)
    
    return food_model, plate_model

def merge_coco_annotations(food_anno_path, plate_anno_path):
    """Merge food and plate COCO annotations with correct ID handling"""
    # Load both annotation files
    with open(food_anno_path, 'r') as f:
        food_anno = json.load(f)
    with open(plate_anno_path, 'r') as f:
        plate_anno = json.load(f)
    
    # Create merged annotation structure
    merged_anno = {
        "info": {
            "year": datetime.now().strftime("%Y"),
            "version": "1",
            "description": "Food and plate Coco",
            "contributor": "Paul",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        },
        "licenses": [
            {
                "id": 1,
                "url": "",
                "name": "CC BY 4.0"
            }
        ],
        "categories": []
    }
    
    # Combine categories with proper supercategory
    category_id_map = {}
    next_category_id = 0
    
    # Add food categories first
    for cat in food_anno["categories"]:
        next_category_id += 1
        new_cat = {
            "id": next_category_id,
            "name": cat["name"],
            "supercategory": "food"
        }
        category_id_map[("food", cat["id"])] = next_category_id
        merged_anno["categories"].append(new_cat)
    
    # Add plate category
    next_category_id += 1
    plate_category = {
        "id": next_category_id,
        "name": "plate",
        "supercategory": "food"
    }
    category_id_map[("plate", 1)] = next_category_id
    merged_anno["categories"].append(plate_category)
    
    # Add image information
    # Use the first image from food annotations as base
    original_filename = food_anno["images"][0]["file_name"]
    # Replace the last dot with underscore
    modified_filename = original_filename.replace('.', '_')
    
    merged_anno["images"] = [
        {
            "id": 0,
            "license": 1,
            "file_name": modified_filename,  # Using modified filename with underscore
            "height": food_anno["images"][0]["height"],
            "width": food_anno["images"][0]["width"],
            "date_captured": food_anno["images"][0]["date_captured"]
        }
    ]
    
    # Combine annotations
    merged_anno["annotations"] = []
    next_anno_id = 0
    
    # Add food annotations
    for anno in food_anno["annotations"]:
        next_anno_id += 1
        new_anno = anno.copy()
        new_anno["id"] = next_anno_id
        new_anno["image_id"] = 0  # Match with image ID
        new_anno["category_id"] = category_id_map[("food", anno["category_id"])]
        merged_anno["annotations"].append(new_anno)
    
    # Add plate annotations
    for anno in plate_anno["annotations"]:
        next_anno_id += 1
        new_anno = anno.copy()
        new_anno["id"] = next_anno_id
        new_anno["image_id"] = 0  # Match with image ID
        new_anno["category_id"] = category_id_map[("plate", anno["category_id"])]
        merged_anno["annotations"].append(new_anno)
    
    # Save merged annotations with correct filename
    final_output_dir = r"D:\csrp2\bytemi\data\segmented"
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, "_annotations.coco.json")
    
    with open(final_output_path, 'w') as f:
        json.dump(merged_anno, f, indent=2)

def process_image(image_path, food_model, plate_model, return_dir):
    """Process a single image with both models and save results"""
    # Create return directory if it doesn't exist
    os.makedirs(return_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process with food detection
    food_detection_script = os.path.join(os.path.dirname(image_path), "detect_food.py")
    if os.path.exists(food_detection_script):
        os.system(f"python {food_detection_script} {image_path}")
    
    # Process with plate detection
    plate_detection_script = os.path.join(os.path.dirname(image_path), "Plateonlydetect.py")
    if os.path.exists(plate_detection_script):
        os.system(f"python {plate_detection_script}")
    
    # Merge annotations
    food_anno_path = os.path.join(return_dir, f"{base_name}.coco.json")
    plate_anno_path = os.path.join(return_dir, "annotations.json")
    merged_anno_path = os.path.join(return_dir, f"{base_name}.merged.json")
    
    if os.path.exists(food_anno_path) and os.path.exists(plate_anno_path):
        merge_coco_annotations(food_anno_path, plate_anno_path, merged_anno_path)
        print(f"Saved annotations to: '_annotations.coco.json' ")
    else:
        print("Warning: One or both annotation files not found")

def food_and_plate_coco():
    # Set up paths
    ROOT_DIR = os.path.abspath(".")
    FOOD_WEIGHTS = os.path.join(ROOT_DIR, "logs", "food20241024T0144", "mask_rcnn_food_0300.h5")
    PLATE_WEIGHTS = os.path.join(ROOT_DIR, "logs", "mask_rcnn_plate_final.h5")
    TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")
    RETURN_DIR = os.path.join(ROOT_DIR, "return")
    
    # Load models
    food_model, plate_model = load_models(FOOD_WEIGHTS, PLATE_WEIGHTS)
    
    # Process images
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.isfile(image_path):
            process_image(image_path, food_model, plate_model, RETURN_DIR)
        else:
            print(f"Error: Cannot find image file: {image_path}")
    else:
        if not os.listdir(TEST_IMAGES_DIR):
            print(f"Please add images to {TEST_IMAGES_DIR}")
            return
            
        for filename in os.listdir(TEST_IMAGES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"\nProcessing {filename}...")
                image_path = os.path.join(TEST_IMAGES_DIR, filename)
                process_image(image_path, food_model, plate_model, RETURN_DIR)

# if __name__ == "__main__":
#     main()