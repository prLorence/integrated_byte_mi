import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import json
import datetime
import pycocotools.mask as mask_util

# Configuration class for the model
class PlateInferenceConfig(Config):
    NAME = "plate"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + plate
    
    DETECTION_MIN_CONFIDENCE = 0.5
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = 0.33
    POST_NMS_ROIS_INFERENCE = 2000
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    USE_MINI_MASK = False
    
    def __init__(self):
        super().__init__()
        self.BACKBONE = "resnet50"
        self.DETECTION_NMS_THRESHOLD = 0.3

def mask_to_polygon(mask):
    """Convert binary mask to polygon points"""
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Get the largest contour (main object)
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour to reduce number of points
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert contour to the format required by COCO
    polygon = []
    for point in approx_contour:
        x, y = point[0]
        polygon.extend([float(x), float(y)])
    
    return polygon

def detect_plates(image_path, model, save_dir=None, coco_annotations=None, image_id=None):
    """
    Detect plates and save results in COCO format with polygon segmentation
    """
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Run detection
    try:
        results = model.detect([image], verbose=1)[0]
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        print(f"Found {len(results['scores'])} objects")
        
        # Add image info to COCO annotations
        if coco_annotations is not None:
            image_info = {
                'id': image_id,
                'file_name': os.path.basename(image_path),
                'width': width,
                'height': height,
                'date_captured': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            coco_annotations['images'].append(image_info)
            
            # Process each detection
            for i in range(len(results['scores'])):
                # Get bbox coordinates
                y1, x1, y2, x2 = results['rois'][i]
                bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]  # COCO format: [x,y,width,height]
                
                # Convert mask to polygon points
                binary_mask = results['masks'][:, :, i].astype(np.uint8)
                polygon = mask_to_polygon(binary_mask)
                
                # Calculate area using contour area for more accuracy
                area = cv2.contourArea(np.array(polygon).reshape(-1, 2).astype(np.int32))
                
                annotation = {
                    'id': len(coco_annotations['annotations']),
                    'image_id': image_id,
                    'category_id': 1,  # Assuming 1 is the category ID for plates
                    'segmentation': [polygon],  # COCO format expects a list of polygons
                    'area': float(area),
                    'bbox': bbox,
                    'iscrowd': 0,
                    'score': float(results['scores'][i])
                }
                coco_annotations['annotations'].append(annotation)
        
        # Visualization code
        if save_dir:
            fig, ax = plt.subplots(1, figsize=(16, 16))
            output = image.copy()
            
            for i in range(len(results['scores'])):
                mask = results['masks'][:, :, i]
                color = np.random.rand(3)
                
                # Apply mask visualization
                for c in range(3):
                    output[:, :, c] = np.where(mask == 1,
                                             output[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                                             output[:, :, c])
                
                # Draw bounding box
                y1, x1, y2, x2 = results['rois'][i]
                cv2.rectangle(output, (x1, y1), (x2, y2), color=tuple(color * 255), thickness=2)
                
                # Add label
                label = f"Plate {results['scores'][i]:.2f}"
                cv2.putText(output, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, tuple(color * 255), 2)
                
                # Draw polygon points
                if coco_annotations is not None:
                    polygon = np.array(coco_annotations['annotations'][-1]['segmentation'][0]).reshape(-1, 2)
                    cv2.polylines(output, [polygon.astype(np.int32)], True, tuple(color * 255), 2)
            
            ax.imshow(output)
            ax.axis('off')
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f"result_{os.path.basename(image_path)}")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
    except Exception as e:
        print(f"Error during detection or annotation: {str(e)}")
        raise e

def plate_detect():
    # Directory setup
    ROOT_DIR = os.path.abspath(".")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_plate_final.h5")
    
    # Initialize COCO format annotations
    coco_annotations = {
        'info': {
            'description': 'Plate Detection Dataset',
            'version': '1.0',
            'year': 2024,
            'contributor': 'Auto-generated',
            'date_created': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'licenses': [{
            'url': 'N/A',
            'id': 1,
            'name': 'N/A'
        }],
        'categories': [{
            'id': 1,
            'name': 'plate',
            'supercategory': 'object'
        }],
        'images': [],
        'annotations': []
    }
    
    # Verify weights file exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return
    
    # Initialize model
    print("\nInitializing model...")
    config = PlateInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=MODEL_DIR)
    
    # Load weights
    print("\nLoading weights...")
    try:
        model.load_weights(WEIGHTS_PATH, by_name=True)
        print("Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return
    
    # Setup directories
    test_dir = os.path.join(ROOT_DIR, "test_images")
    output_dir = os.path.join(ROOT_DIR, "test_results")
    return_dir = r"D:\csrp2\bytemi\return"
    
    # Create output directories if they don't exist
    for directory in [output_dir, return_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process each image
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(test_dir, filename)
        detect_plates(image_path, model, output_dir, coco_annotations, image_id=idx)
    
    # Save COCO annotations
    annotations_path = os.path.join(return_dir, 'annotations.json')
    try:
        with open(annotations_path, 'w') as f:
            json.dump(coco_annotations, f, indent=2)
        print(f"\nSaved COCO annotations to: {annotations_path}")
    except Exception as e:
        print(f"Error saving annotations: {str(e)}")

# if __name__ == "__main__":
#     main()