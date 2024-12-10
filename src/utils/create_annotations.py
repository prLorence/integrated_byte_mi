from datetime import datetime
from typing import Dict, List
import cv2 
import numpy as np

def create_annotations(results: Dict, image_array: np.ndarray, image_path: str, class_names: List[str], is_food: bool = True) -> Dict:
    """Create COCO format annotations from detection results."""
    coco_data = {
        "info": {
            "year": datetime.now().strftime("%Y"),
            "version": "1",
            "description": "Food Detection Results" if is_food else "Plate Detection Results",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [{
            "id": 1,
            "width": image_array.shape[1],
            "height": image_array.shape[0],
            "file_name": str(image_path)
        }],
        "categories": [],
        "annotations": []
    }

    # Add categories
    if is_food:
        for idx, name in enumerate(class_names, 1):
            coco_data["categories"].append({
                "id": idx,
                "name": name,
                "supercategory": "food"
            })
    else:
        coco_data["categories"].append({
            "id": 1,
            "name": "plate",
            "supercategory": "plate"
        })

    # Add annotations
    for i in range(len(results['rois'])):
        y1, x1, y2, x2 = results['rois'][i]
        
        # Convert mask to polygon if masks exist
        segmentation = []
        if 'masks' in results and results['masks'].shape[-1] > i:
            mask = results['masks'][:, :, i]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                segmentation.append(contour.flatten().tolist())

        annotation = {
            "id": i + 1,
            "image_id": 1,
            "category_id": int(results['class_ids'][i]) if is_food else 1,
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "area": float((x2-x1) * (y2-y1)),
            "segmentation": segmentation,
            "iscrowd": 0,
            "score": float(results['scores'][i])
        }
        
        coco_data["annotations"].append(annotation)

    return coco_data