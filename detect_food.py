import os
import sys
import json
import numpy as np
import tensorflow as tf
from mrcnn import model as modellib
from mrcnn.config import Config
from PIL import Image
from datetime import datetime
from skimage import measure

class FoodConfig(Config):
    NAME = "food"
    
    meta_path = os.path.join(os.path.abspath("."), "food-recognition-dataset", "meta.json")
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
        NUM_CLASSES = 1 + len(meta_data['classes'])
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class InferenceConfig(FoodConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.15
    DETECTION_MAX_INSTANCES = 25
    DETECTION_NMS_THRESHOLD = 0.25
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_NMS_THRESHOLD = 0.6
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

def get_food_classes(meta_path):
    try:
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        classes = ['BG']
        classes.extend([cls['title'] for cls in meta_data['classes']])
        return classes
    except Exception as e:
        print(f"Error loading meta.json: {e}")
        return None

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)

def filter_similar_detections(results, iou_threshold=0.25, score_threshold=0.15):
    if len(results['rois']) == 0:
        return results
    
    indices = np.argsort(results['scores'])[::-1]
    keep_indices = []
    
    def calculate_iou(box1, box2):
        y1 = max(box1[0], box2[0])
        x1 = max(box1[1], box2[1])
        y2 = min(box1[2], box2[2])
        x2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    for idx in indices:
        if results['scores'][idx] < score_threshold:
            continue
        
        overlaps = False
        current_box = results['rois'][idx]
        
        for kept_idx in keep_indices:
            kept_box = results['rois'][kept_idx]
            iou = calculate_iou(current_box, kept_box)
            
            if iou > iou_threshold:
                overlaps = True
                break
        
        if not overlaps:
            keep_indices.append(idx)
    
    filtered_results = {
        'rois': results['rois'][keep_indices],
        'class_ids': results['class_ids'][keep_indices],
        'scores': results['scores'][keep_indices],
        'masks': results['masks'][:, :, keep_indices] if results['masks'].size > 0 else results['masks']
    }
    
    return filtered_results

def convert_to_coco_format(image_path, results, class_names):
    """Convert detection results to proper COCO format with segmentation points"""
    image = Image.open(image_path)
    width, height = image.size
    
    coco_output = {
        "info": {
            "description": "Food Detection Results",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "contributor": "Mask RCNN Food Detector"
        },
        "licenses": [{"url": "N/A", "id": 1, "name": "N/A"}],
        "images": [{
            "id": 1,
            "width": int(width),
            "height": int(height),
            "file_name": os.path.basename(image_path),
            "license": 1,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }],
        "categories": [],
        "annotations": []
    }
    
    detected_class_ids = set([int(x) for x in results['class_ids']])
    for idx in detected_class_ids:
        if idx == 0:  # Skip background class
            continue
        coco_output["categories"].append({
            "id": int(idx - 1),
            "name": class_names[idx],
            "supercategory": "food"
        })
    
    annotation_id = 1
    
    for i in range(len(results['rois'])):
        y1, x1, y2, x2 = [float(x) for x in results['rois'][i]]
        bbox_width = float(x2 - x1)
        bbox_height = float(y2 - y1)
        
        if results['masks'].size > 0:
            mask = results['masks'][:, :, i].astype(np.uint8)
            contours = measure.find_contours(mask, 0.5)
            segmentations = []
            
            for contour in contours:
                contour = np.fliplr(contour)
                contour = [float(x) for x in contour.flatten()]
                if len(contour) > 4:
                    if contour[0] != contour[-2] or contour[1] != contour[-1]:
                        contour.extend([float(contour[0]), float(contour[1])])
                    segmentations.append(contour)
            
            if not segmentations:
                segmentations = [[
                    float(x1), float(y1),
                    float(x2), float(y1),
                    float(x2), float(y2),
                    float(x1), float(y2),
                    float(x1), float(y1)
                ]]
        else:
            segmentations = [[
                float(x1), float(y1),
                float(x2), float(y1),
                float(x2), float(y2),
                float(x1), float(y2),
                float(x1), float(y1)
            ]]
        
        if results['masks'].size > 0:
            area = float(np.sum(mask))
        else:
            area = float(bbox_width * bbox_height)
        
        annotation = {
            "id": int(annotation_id),
            "image_id": 1,
            "category_id": int(results['class_ids'][i] - 1),
            "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
            "area": float(area),
            "segmentation": segmentations,
            "iscrowd": 0,
            "score": float(results['scores'][i])
        }
        
        coco_output["annotations"].append(annotation)
        annotation_id += 1
    
    return coco_output

def detect_food(image_path, model_dir, meta_path):
    print("Starting food detection...")
    
    class_names = get_food_classes(meta_path)
    if not class_names:
        print("Failed to load class names!")
        return

    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    
    model_path = r"D:\csrp2\bytemi\logs\food20241024T0144\mask_rcnn_food_0300.h5"
    print(f"Loading weights from {model_path}")
    model.load_weights(model_path, by_name=True)
    
    try:
        image = Image.open(image_path)
        image = preprocess_image(image)
        print(f"Image shape: {image.shape}")
        
        results = model.detect([image], verbose=1)[0]
        filtered_results = filter_similar_detections(
            results, 
            iou_threshold=0.25,
            score_threshold=0.15
        )
        
        sort_idx = np.argsort(filtered_results['scores'])[::-1]
        filtered_results = {
            'rois': filtered_results['rois'][sort_idx],
            'class_ids': filtered_results['class_ids'][sort_idx],
            'scores': filtered_results['scores'][sort_idx],
            'masks': filtered_results['masks'][:, :, sort_idx] if filtered_results['masks'].size > 0 else filtered_results['masks']
        }
        
        print("\nDetected Foods:")
        for i in range(len(filtered_results['class_ids'])):
            class_id = filtered_results['class_ids'][i]
            score = filtered_results['scores'][i]
            print(f"- {class_names[class_id]} (confidence: {score:.3f})")
        
        # Save COCO annotations
        coco_data = convert_to_coco_format(image_path, filtered_results, class_names)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)), "return")
        os.makedirs(return_dir, exist_ok=True)
        
        coco_output_path = os.path.join(return_dir, f"{base_name}.coco.json")
        with open(coco_output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        print(f"Saved COCO annotations to: {coco_output_path}")
        
        return filtered_results, class_names
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def start():
    ROOT_DIR = os.path.abspath(".")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    META_PATH = os.path.join(ROOT_DIR, "food-recognition-dataset", "meta.json")
    TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")
    
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.isfile(image_path):
            detect_food(image_path, MODEL_DIR, META_PATH)
        else:
            print(f"Error: Cannot find image file: {image_path}")
    else:
        if not os.listdir(TEST_IMAGES_DIR):
            print(f"Please add images to {TEST_IMAGES_DIR}")
            return
            
        for filename in os.listdir(TEST_IMAGES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"\nProcessing {filename}...")
                detect_food(
                    os.path.join(TEST_IMAGES_DIR, filename),
                    MODEL_DIR,
                    META_PATH
                )

if __name__ == '__main__':
    start()


# def food_detection():
#     ROOT_DIR = os.path.abspath(".")
#     MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#     META_PATH = os.path.join(ROOT_DIR, "food-recognition-dataset", "meta.json")
#     TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")
    
#     os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
#     if len(sys.argv) > 1:
#         image_path = sys.argv[1]
#         if os.path.isfile(image_path):
#             detect_food(image_path, MODEL_DIR, META_PATH)
#         else:
#             print(f"Error: Cannot find image file: {image_path}")
#     else:
#         if not os.listdir(TEST_IMAGES_DIR):
#             print(f"Please add images to {TEST_IMAGES_DIR}")
#             return
            
#         for filename in os.listdir(TEST_IMAGES_DIR):
#             if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 print(f"\nProcessing {filename}...")
#                 detect_food(
#                     os.path.join(TEST_IMAGES_DIR, filename),
#                     MODEL_DIR,
#                     META_PATH
#                 )

# # if __name__ == '__main__':
# #     main()