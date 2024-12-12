import numpy as np
from typing import Dict, Tuple, Optional
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeCalculator:
    def __init__(self, camera_height: float = 33.0, plate_diameter: float = 25.5):
        """Initialize the volume calculator with camera parameters.
        
        Args:
            camera_height: Height of camera in cm
            plate_diameter: Diameter of reference plate in cm
        """
        self.camera_height = camera_height
        self.plate_diameter = plate_diameter
        self.plate_height = 0.7
        self.CM3_TO_CUPS = 0.0338140225
        self.MAX_FOOD_HEIGHT = 5.0

    def calculate_plate_reference(self, depth_map: np.ndarray,
                                plate_mask: np.ndarray,
                                intrinsic_params: Dict) -> Dict[str, float]:
        """Calculate reference measurements using the plate.
        
        Args:
            depth_map: Raw depth values
            plate_mask: Binary mask of plate region
            intrinsic_params: Camera intrinsic parameters
            
        Returns:
            Dict containing scale_factor, plate_height, and pixel_to_cm
        """
        try:
            if not np.any(plate_mask):
                raise ValueError("Empty plate mask")

            # Convert mask to uint8 for OpenCV operations
            plate_mask_uint8 = plate_mask.astype(np.uint8)
            
            # Get plate depths
            plate_depths = depth_map[plate_mask > 0]
            if len(plate_depths) == 0:
                raise ValueError("No depth values in plate region")

            # Calculate depth scaling
            depth_range = max(np.ptp(depth_map), 1)
            depth_scale = (self.camera_height - self.plate_height) / depth_range
            plate_depths_cm = plate_depths * depth_scale

            # Get plate dimensions using contour
            contours, _ = cv2.findContours(plate_mask_uint8, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("No plate contour found")
                
            plate_contour = max(contours, key=cv2.contourArea)
            plate_area = cv2.contourArea(plate_contour)
            
            # Calculate pixel to cm conversion
            measured_diameter_pixels = 2 * np.sqrt(plate_area / np.pi)
            pixel_to_cm = self.plate_diameter / measured_diameter_pixels

            # Calculate plate height using central region
            center_mask = np.zeros_like(plate_mask_uint8)
            moments = cv2.moments(plate_mask_uint8)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                radius = int(measured_diameter_pixels * 0.3)
                cv2.circle(center_mask, (cx, cy), radius, 1, -1)
                center_depths = depth_map[center_mask > 0]
                plate_height = np.median(center_depths * depth_scale) if len(center_depths) > 0 else np.median(plate_depths_cm)
            else:
                plate_height = np.median(plate_depths_cm)

            # Calculate scale factor
            measured_area = plate_area * (pixel_to_cm ** 2)
            actual_area = np.pi * (self.plate_diameter/2)**2
            scale_factor = actual_area / max(measured_area, 0.001)
            scale_factor = np.clip(scale_factor, 0.5, 2.0)

            logger.info(f"Plate Reference Calculations:")
            logger.info(f"Depth scale: {depth_scale:.6f}")
            logger.info(f"Pixel to cm: {pixel_to_cm:.6f}")
            logger.info(f"Plate height: {plate_height:.4f} cm")
            logger.info(f"Scale factor: {scale_factor:.4f}")

            return {
                'scale_factor': float(scale_factor),
                'plate_height': float(plate_height),
                'pixel_to_cm': float(pixel_to_cm)
            }

        except Exception as e:
            logger.error(f"Error in plate calibration: {str(e)}")
            # Return reasonable defaults
            return {
                'scale_factor': 1.0,
                'plate_height': self.camera_height - 1.0,
                'pixel_to_cm': 0.1
            }

    def _analyze_shape(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze object shape characteristics."""
        try:
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {'solidity': 0.8, 'aspect_ratio': 1.0}

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.8

            # Calculate aspect ratio
            rect = cv2.minAreaRect(contour)
            width = min(rect[1]) if rect[1][0] != 0 and rect[1][1] != 0 else 1
            length = max(rect[1]) if rect[1][0] != 0 and rect[1][1] != 0 else 1
            aspect_ratio = length / width

            return {
                'solidity': float(solidity),
                'aspect_ratio': float(aspect_ratio)
            }

        except Exception as e:
            logger.warning(f"Shape analysis failed: {str(e)}")
            return {'solidity': 0.8, 'aspect_ratio': 1.0}
    def calculate_volume(self, depth_map: np.ndarray,
                        mask: np.ndarray,
                        plate_height: float,
                        intrinsic_params: Dict,
                        calibration: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate volume with corrected scaling"""
        try:
            if not np.any(mask):
                raise ValueError("Empty mask provided")

            # Get masked depths
            masked_depths = depth_map[mask > 0]
            if len(masked_depths) == 0:
                raise ValueError("No depth values in mask")

            # Calculate depth range and scale
            # Assuming depth values are in the range [306, 345] for 33cm camera height
            total_depth_range = self.camera_height  # 33cm total range
            raw_range = np.max(depth_map) - np.min(depth_map)
            depth_scale = total_depth_range / raw_range

            # Convert depths to centimeters
            depths_cm = masked_depths * depth_scale / 10  # Scale down by factor of 10
            rel_heights = np.abs(depths_cm - np.median(depths_cm))
            heights = np.clip(rel_heights, 0, self.MAX_FOOD_HEIGHT)

            # Log depth statistics
            logger.info(f"Depth Analysis:")
            logger.info(f"Raw depth range: [{np.min(masked_depths)}, {np.max(masked_depths)}]")
            logger.info(f"Depth scale: {depth_scale:.6f}")
            logger.info(f"Heights range: [{np.min(heights):.2f}, {np.max(heights):.2f}] cm")

            # Analyze shape
            shape_metrics = self._analyze_shape(mask)
            solidity = shape_metrics['solidity']
            aspect_ratio = shape_metrics['aspect_ratio']

            # Determine object type and factors
            if solidity > 0.85:  
                area_factor = 0.95
                height_factor = 0.7  # Reduced from 1.1
                volume_factor = 0.9  # Reduced from 1.0
                max_expected_height = 3.0
            elif aspect_ratio > 2.0:  # Elongated (cucumber)
                area_factor = 0.85
                height_factor = 0.6  # Reduced from 1.0
                volume_factor = 0.8  # Reduced from 0.9
                max_expected_height = 2.5
            else:  # Spread (rice)
                area_factor = 0.75
                height_factor = 0.5  # Reduced from 0.9
                volume_factor = 0.7  # Reduced from 0.85
                max_expected_height = 2.0

            # Scale heights to expected range
            height_scale = max_expected_height / np.max(heights) if np.max(heights) > 0 else 1.0
            heights *= height_scale

            # Calculate base area with pixel size correction
            pixel_to_cm = calibration.get('pixel_to_cm', 0.1)
            base_area = np.sum(mask) * (pixel_to_cm ** 2) * area_factor
            base_area *= 0.2  # Global area correction factor

            # Calculate volume using weighted heights
            sorted_heights = np.sort(heights)
            weights = np.linspace(0.7, 1.0, len(sorted_heights))  # More conservative weighting
            avg_height = np.average(sorted_heights, weights=weights) * height_factor

            # Calculate initial volume
            volume_cm3 = base_area * avg_height * volume_factor
            
            # Apply calibration
            if calibration and 'scale_factor' in calibration:
                volume_cm3 *= np.clip(calibration['scale_factor'], 0.5, 1.5)

            # Convert to cups with reasonable bounds
            volume_cm3 = np.clip(volume_cm3, 0.1, 300.0)
            volume_cups = volume_cm3 * self.CM3_TO_CUPS

            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(len(masked_depths), heights, shape_metrics)

            logger.info(f"Volume Calculation:")
            logger.info(f"Object type: {'Compact' if solidity > 0.85 else 'Elongated' if aspect_ratio > 2.0 else 'Spread'}")
            logger.info(f"Base area: {base_area:.2f} cm²")
            logger.info(f"Average height: {avg_height:.4f} cm")
            logger.info(f"Volume: {volume_cm3:.2f} cm³ ({volume_cups:.2f} cups)")

            return {
                'volume_cm3': float(volume_cm3),
                'volume_cups': float(volume_cups),
                'uncertainty_cm3': float(uncertainty * volume_cm3),
                'uncertainty_cups': float(uncertainty * volume_cups),
                'base_area_cm2': float(base_area),
                'avg_height_cm': float(avg_height),
                'max_height_cm': float(np.max(heights))
            }

        except Exception as e:
            logger.error(f"Error in volume calculation: {str(e)}")
            raise

    def _calculate_uncertainty(self, num_points: int, heights: np.ndarray, 
                            shape_metrics: Dict[str, float]) -> float:
        """Calculate uncertainty with refined factors"""
        try:
            # Base uncertainty
            if shape_metrics['solidity'] > 0.85:  # Compact
                base = 0.2
            elif shape_metrics['aspect_ratio'] > 2.0:  # Elongated
                base = 0.25
            else:  # Spread
                base = 0.3
            
            # Point density factor
            point_factor = min(1.0, np.sqrt(300 / max(num_points, 1))) * 0.1
            
            # Height variation factor
            height_std = np.std(heights) if len(heights) > 0 else 0
            height_mean = np.mean(heights) if len(heights) > 0 else 1
            height_factor = min(height_std / max(height_mean, 0.001), 0.2)
            
            total = base + point_factor + height_factor
            return np.clip(total, 0.15, 0.4)

        except Exception as e:
            logger.warning(f"Uncertainty calculation failed: {str(e)}")
            return 0.3

