def calculate_volume(self, depth_map: np.ndarray,
                    mask: np.ndarray,
                    plate_height: float,
                    intrinsic_params: Dict,
                    calibration: Optional[Dict] = None) -> Dict[str, float]:
    """Calculate volume with corrected scaling factors."""
    try:
        if not np.any(mask):
            raise ValueError("Empty mask provided")

        # Validate depth data
        depth_stats = self._validate_depth_map(depth_map, mask)
        if not depth_stats['is_valid']:
            logger.warning(f"Using fallback calculation: {depth_stats['reason']}")
            return self._calculate_volume_from_area(mask, calibration)

        # Get masked depths and apply smoothing
        masked_depths = depth_map[mask > 0]
        if len(masked_depths) == 0:
            return self._calculate_volume_from_area(mask, calibration)

        masked_depths = self._smooth_depths(masked_depths)

        # Calculate depth scale (corrected for ARCore depth values)
        depth_range = np.ptp(depth_map)
        raw_min = np.min(depth_map[depth_map > 0])
        depth_scale = 0.1  # Fixed scale factor for pseudo depth
        
        # Convert depths to relative heights
        heights = np.abs(masked_depths - raw_min) * depth_scale
        heights = np.clip(heights, 0, self.MAX_FOOD_HEIGHT)

        # Get shape characteristics
        shape_metrics = self._analyze_shape(mask)
        solidity = shape_metrics['solidity']
        aspect_ratio = shape_metrics['aspect_ratio']

        # Determine object type and adjustment factors (reduced factors)
        if solidity > 0.85:  # Compact (egg)
            area_factor = 0.95
            height_factor = 0.6
            volume_factor = 0.9
            max_expected_height = 3.0
        elif aspect_ratio > 2.0:  # Elongated (cucumber)
            area_factor = 0.85
            height_factor = 0.5
            volume_factor = 0.7
            max_expected_height = 2.5
        else:  # Spread (rice)
            area_factor = 0.8
            height_factor = 0.4
            volume_factor = 1.1
            max_expected_height = 2.0

        # Scale heights to expected range
        height_scale = max_expected_height / np.max(heights) if np.max(heights) > 0 else 1.0
        heights *= height_scale

        # Calculate base area with reduced scaling
        pixel_to_cm = calibration.get('pixel_to_cm', 0.1)
        base_area = np.sum(mask) * (pixel_to_cm ** 2) * area_factor
        base_area *= 0.2  # Reduced area scaling

        # Calculate volume using weighted heights
        sorted_heights = np.sort(heights)
        weights = np.linspace(0.8, 1.0, len(sorted_heights))
        avg_height = np.average(sorted_heights, weights=weights) * height_factor

        # Calculate volume
        volume_cm3 = base_area * avg_height * volume_factor
        
        # Apply calibration with tighter bounds
        if calibration and 'scale_factor' in calibration:
            scale_factor = np.clip(calibration['scale_factor'], 0.5, 1.5)
            volume_cm3 *= scale_factor

        # Convert to cups with tighter bounds
        volume_cm3 = np.clip(volume_cm3, 0.1, 200.0)
        volume_cups = volume_cm3 * self.CM3_TO_CUPS

        # Calculate uncertainty and stability
        stability_score = self._assess_measurement_stability(
            masked_depths, heights, shape_metrics
        )
        uncertainty = self._calculate_uncertainty(
            len(masked_depths), heights, shape_metrics, stability_score
        )

        logger.info(f"Volume Calculation Results:")
        logger.info(f"Object type: {'Compact' if solidity > 0.85 else 'Elongated' if aspect_ratio > 2.0 else 'Spread'}")
        logger.info(f"Base area: {base_area:.2f} cm²")
        logger.info(f"Average height: {avg_height:.4f} cm")
        logger.info(f"Volume: {volume_cm3:.2f} cm³ ({volume_cups:.2f} cups)")
        logger.info(f"Stability Score: {stability_score:.3f}")

        return {
            'volume_cm3': float(volume_cm3),
            'volume_cups': float(volume_cups),
            'uncertainty_cm3': float(uncertainty * volume_cm3),
            'uncertainty_cups': float(uncertainty * volume_cups),
            'base_area_cm2': float(base_area),
            'avg_height_cm': float(avg_height),
            'max_height_cm': float(np.max(heights)),
            'stability_score': float(stability_score)
        }

    except Exception as e:
        logger.error(f"Error in volume calculation: {str(e)}")
        return self._calculate_volume_from_area(mask, calibration)
