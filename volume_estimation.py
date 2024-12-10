
import json
from pathlib import Path
from src.preprocessing.preprocessing import PreprocessingPipeline
from src.reconstruction.volume_calculator import VolumeCalculator
from src.utils.visualization_3d import Visualizer3D
from src.utils.logging_utils import setup_logging
from module import run_script

def process_frames(config, frame_id):
    pipeline = PreprocessingPipeline(config)
    calc = VolumeCalculator(
        camera_height=config['camera_height'],
        plate_diameter=config['plate_diameter']
    )

    result = pipeline.process_single_image(frame_id)
    plate_data = next(
        (data for name, data in result['processed_objects'].items() if name == 'plate'),
        None
    )
    if plate_data is None:
        raise ValueError("No plate found in processed objects")
    
    calibration = calc.calculate_plate_reference(
        depth_map=result['depth'],
        plate_mask=plate_data['mask'],
        intrinsic_params=result['intrinsic_params']
    )
    plate_height = calibration['plate_height']
    
    volume_results = {}
    for obj_name, obj_data in result['processed_objects'].items():
        if obj_name == 'plate':
            continue
        volume_data = calc.calculate_volume(
            depth_map=result['depth'],
            mask=obj_data['mask'],
            plate_height=plate_height,
            intrinsic_params=result['intrinsic_params'],
            calibration=calibration
        )
        volume_results[obj_name] = volume_data

    return {
        "frame_id": frame_id,
        "volumes": [
            {
                "object_name": obj_name,
                "volume_cups": volume_data['volume_cups'],
                "uncertainty_cups": volume_data['uncertainty_cups']
            }
            for obj_name, volume_data in volume_results.items()
        ]
    }