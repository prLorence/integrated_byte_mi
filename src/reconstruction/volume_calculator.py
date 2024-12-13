def calculate_plate_reference(self, depth_map: np.ndarray,
                            plate_mask: np.ndarray,
                            intrinsic_params: Dict) -> Dict[str, float]:
    """Calculate plate reference measurements with robust handling of pseudo depth data."""
    try:
        # Validate depth data
        depth_stats = self._validate_depth_map(depth_map, plate_mask)
        if not depth_stats['is_valid']:
            logger.warning(f"Unreliable plate depth data: {depth_stats['reason']}")

        # Convert mask to uint8 for OpenCV operations
        plate_mask_uint8 = plate_mask.astype(np.uint8)
        
        # Get plate depths with smoothing
        plate_depths = depth_map[plate_mask > 0]
        if len(plate_depths) == 0:
            raise ValueError("No depth values in plate region")
            
        plate_depths = self._smooth_depths(plate_depths)

        # Calculate depth scale
        depth_range = np.ptp(depth_map)
        if depth_range < 1:
            logger.warning("Very small depth range, using default scale")
            depth_scale = 0.1
        else:
            depth_scale = (self.camera_height - self.plate_height) / depth_range
            
        plate_depths_cm = plate_depths * depth_scale / 10

        # Get plate dimensions using contour
        contours, _ = cv2.findContours(plate_mask_uint8, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No plate contour found")
            
        plate_contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), radius = cv2.minEnclosingCircle(plate_contour)
        
        # Calculate pixel to cm conversion
        pixel_to_cm = self.plate_diameter / (2 * radius)

        # Calculate plate height using central region
        center_mask = np.zeros_like(plate_mask_uint8)
        cv2.circle(center_mask, (int(center_x), int(center_y)), 
                  int(radius * 0.3), 1, -1)
        
        center_depths = depth_map[center_mask > 0]
        if len(center_depths) > 0:
            center_depths = self._smooth_depths(center_depths)
            plate_height = np.median(center_depths * depth_scale / 10)
        else:
            plate_height = np.median(plate_depths_cm)

        # Calculate scale factor
        measured_area = cv2.contourArea(plate_contour) * (pixel_to_cm ** 2)
        actual_area = np.pi * (self.plate_diameter/2)**2
        scale_factor = actual_area / max(measured_area, 0.001)
        scale_factor = np.clip(scale_factor, 0.5, 1.5)

        # Calculate stability score for plate measurements
        stability_score = self._assess_measurement_stability(
            plate_depths,
            np.full_like(plate_depths, plate_height),
            {'solidity': 1.0, 'aspect_ratio': 1.0}
        )

        logger.info(f"Plate Reference Calculations:")
        logger.info(f"Depth scale: {depth_scale:.6f}")
        logger.info(f"Pixel to cm: {pixel_to_cm:.6f}")
        logger.info(f"Plate height: {plate_height:.4f} cm")
        logger.info(f"Scale factor: {scale_factor:.4f}")
        logger.info(f"Stability score: {stability_score:.3f}")

        return {
            'scale_factor': float(scale_factor),
            'plate_height': float(plate_height),
            'pixel_to_cm': float(pixel_to_cm),
            'stability_score': float(stability_score)
        }

    except Exception as e:
        logger.error(f"Error in plate calibration: {str(e)}")
        # Return reasonable defaults
        return {
            'scale_factor': 1.0,
            'plate_height': self.camera_height - 1.0,
            'pixel_to_cm': 0.1,
            'stability_score': 0.5
        }
