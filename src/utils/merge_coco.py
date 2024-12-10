from datetime import datetime
import json


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
    
    # Add image information from food annotations
    merged_anno["images"] = [
        {
            "id": 0,
            "license": 1,
            "file_name": food_anno["images"][0]["file_name"],
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
    
    return merged_anno