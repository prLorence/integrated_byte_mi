import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
import cv2
from pycocotools import mask as maskUtils

class AnnotationVisualizer:
    def __init__(self, annotation_file):
        """Initialize the visualizer with annotation file"""
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup dictionaries
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Create color map for categories
        self.color_map = self._generate_color_map()
    
    def _generate_color_map(self):
        """Generate distinct colors for each category"""
        color_map = {}
        
        # Assign specific color for plate category
        for cat_id, cat in self.categories.items():
            if cat['name'].lower() == 'plate':
                color_map[cat_id] = np.array([0.9, 0.9, 0.9])  # Light gray for plate
        
        # Generate colors for other categories
        for cat_id, cat in self.categories.items():
            if cat_id not in color_map:
                if cat['name'].lower() == 'banana':
                    color_map[cat_id] = np.array([1.0, 0.8, 0.0])  # Yellow for banana
                elif 'fries' in cat['name'].lower():
                    color_map[cat_id] = np.array([0.0, 0.8, 0.0])  # Green for fries
                else:
                    color_map[cat_id] = np.random.rand(3)
        
        return color_map
    
    def decode_rle(self, segmentation, shape):
        """Decode RLE format to binary mask"""
        if isinstance(segmentation, dict):
            if 'counts' in segmentation and 'size' in segmentation:
                try:
                    # Handle compressed RLE format
                    if isinstance(segmentation['counts'], str):
                        rle = {'counts': segmentation['counts'].encode('utf-8'), 'size': segmentation['size']}
                        mask = maskUtils.decode(rle)
                    else:
                        mask = maskUtils.decode(segmentation)
                    return mask.astype(bool)
                except Exception as e:
                    print(f"Error decoding RLE: {e}")
                    return None
        elif isinstance(segmentation, list):
            return self._polygon_to_mask(segmentation, shape)
        return None
    
    def _polygon_to_mask(self, polygons, shape):
        """Convert polygon points to binary mask"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        
        try:
            for polygon in polygons:
                # Ensure we have valid polygon points (must have at least 3 points)
                if len(polygon) < 6:  # Less than 3 points (x,y pairs)
                    continue
                
                # Reshape points to (-1, 2) and convert to int32
                points = np.array(polygon).reshape(-1, 2)
                # Check for valid points (no NaN or Inf)
                if not np.all(np.isfinite(points)):
                    continue
                    
                points = points.astype(np.int32)
                
                # Ensure points are within image boundaries
                points[:, 0] = np.clip(points[:, 0], 0, shape[1]-1)
                points[:, 1] = np.clip(points[:, 1], 0, shape[0]-1)
                
                # Draw the polygon
                cv2.fillPoly(mask, [points], 1)
                
        except Exception as e:
            print(f"Error in polygon to mask conversion: {e}")
            return np.zeros(shape[:2], dtype=bool)
            
        return mask.astype(bool)
    
    def visualize_image(self, image_path, output_path=None, show_labels=True, 
                       show_masks=True, show_boxes=True, alpha=0.4):
        """Visualize annotations for a single image"""
        # Read image
        image = np.array(Image.open(image_path))
        height, width = image.shape[:2]
        
        # Create figure
        plt.figure(figsize=(16, 16))
        plt.imshow(image)
        
        # Find corresponding image in annotations
        image_filename = os.path.basename(image_path)
        image_id = None
        for img in self.coco_data['images']:
            if img['file_name'] == image_filename:
                image_id = img['id']
                break
        
        if image_id is None:
            print(f"No annotations found for image: {image_filename}")
            return
        
        # Filter and sort annotations (plate first, then other objects)
        annotations = [ann for ann in self.coco_data['annotations'] 
                      if ann['image_id'] == image_id or ann['image_id'] == 0]
        
        # Sort so plate is drawn first
        annotations.sort(key=lambda x: 
                        0 if self.categories[x['category_id']]['name'].lower() == 'plate' 
                        else x['area'])
        
        # Create a copy of the image for overlaying masks
        result_image = image.copy().astype(np.float32)
        
        for ann in annotations:
            try:
                category_id = ann['category_id']
                category_name = self.categories[category_id]['name']
                color = self.color_map[category_id]
                
                # Draw mask
                if show_masks and 'segmentation' in ann:
                    mask = self.decode_rle(ann['segmentation'], (height, width))
                    if mask is not None:
                        # Create colored mask
                        for c in range(3):
                            result_image[:, :, c] = np.where(mask,
                                                           result_image[:, :, c] * (1 - alpha) + color[c] * 255 * alpha,
                                                           result_image[:, :, c])
                
                # Draw bounding box
                if show_boxes and 'bbox' in ann:
                    x, y, w, h = [int(coord) for coord in ann['bbox']]
                    rect = Rectangle((x, y), w, h, fill=False, 
                                   edgecolor=color, linewidth=2)
                    plt.gca().add_patch(rect)
                
                # Add label
                if show_labels and 'bbox' in ann:
                    score = ann.get('score', 1.0)
                    label = f"{category_name} {score:.2f}"
                    plt.text(x, y-5, label, color='white', fontweight='bold',
                            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))
            
            except Exception as e:
                print(f"Error processing annotation: {e}")
                continue
        
        # Display the final image with all overlays
        plt.imshow(result_image.astype(np.uint8))
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            print(f"Saved visualization to: {output_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    # Set up directories
    root_dir = os.path.abspath(".")
    test_dir = os.path.join(root_dir, "test_images")
    return_dir = os.path.join(root_dir, "return")
    viz_dir = os.path.join(root_dir, "visualization_results")
    
    # Create visualization directory
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load merged annotations
    merged_annotations_path = os.path.join(return_dir, "_annotations.coco.json")
    if not os.path.exists(merged_annotations_path):
        print(f"Error: Merged annotations file not found at {merged_annotations_path}")
        return
    
    # Initialize visualizer
    visualizer = AnnotationVisualizer(merged_annotations_path)
    
    # Process each image
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"\nProcessing {filename}...")
            image_path = os.path.join(test_dir, filename)
            output_path = os.path.join(viz_dir, f"viz_{filename}")
            
            try:
                visualizer.visualize_image(
                    image_path=image_path,
                    output_path=output_path,
                    show_labels=True,
                    show_masks=True,
                    show_boxes=True,
                    alpha=0.4
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()