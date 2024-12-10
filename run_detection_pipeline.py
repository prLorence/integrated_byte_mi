from datetime import datetime
import json
import logging
import os
from pathlib import Path
from Plateonlydetect import mask_to_polygon
from detect_food import convert_to_coco_format, filter_similar_detections
# from detect_food_with_plate import merge_coco_annotations
from model_singleton import ModelSingleton
import numpy as np
from PIL import Image
from src.utils.merge_coco import merge_coco_annotations
import cv2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_detection_pipeline(image_path: str, model_singleton: ModelSingleton, frame_id: str) -> str:
    """
    Run detection pipeline and create COCO annotations
    
    Args:
        image_path: Path to input image
        model_singleton: Initialized ModelSingleton instance
        frame_id: Frame identifier
        
    Returns:
        str: Path to merged COCO annotations file
    """
    try:
        # Create return directory if it doesn't exist
        return_path = Path("return")
        return_path.mkdir(exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load and process image
        image_array = np.array(Image.open(image_path))
        
        # Run detections
        plate_results = model_singleton.detect_plate(image_array)
        food_results = model_singleton.detect_food(image_array)
        filtered_food_results = filter_similar_detections(
            food_results,
            iou_threshold=0.25,
            score_threshold=0.15
        )

        food_coco_data = convert_to_coco_format(image_path, filtered_food_results, model_singleton.class_names)

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
            binary_mask = plate_results['masks'][:, :, i].astype(np.uint8)
            polygon = mask_to_polygon(binary_mask)
            area = cv2.contourArea(np.array(polygon).reshape(-1, 2).astype(np.int32))

            plate_coco_data["annotations"].append({
                "id": i + 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "area": float(area),
                "segmentation": [polygon],  # Add proper segmentation if needed
                "iscrowd": 0,
                "score": float(plate_results['scores'][i])
        })
        
        # Convert numpy arrays to lists for JSON serialization
        # Save detection results
        food_coco_path = return_path / f"{base_name}.coco.json"
        plate_coco_path = return_path / "annotations.json"

        with open(food_coco_path, 'w') as f:
            json.dump(food_coco_data, f, indent=2, ensure_ascii=False)

        with open(plate_coco_path, 'w') as f:
            json.dump(plate_coco_data, f, indent=2, ensure_ascii=False)

        # Merge annotations
        merged_annotations = merge_coco_annotations(str(food_coco_path), str(plate_coco_path))
        
        # Save merged annotations
        merged_path = return_path / "_annotations.coco.json"
        with open(merged_path, 'w') as f:
            json.dump(merged_annotations, f, indent=2)

        return str(merged_path)

    except Exception as e:
        logger.error(f"Error in detection pipeline: {e}", exc_info=True)
        raise