import tensorflow as tf
from mrcnn import model as modellib
from threading import Lock
from functools import lru_cache
from detect_food import InferenceConfig, get_food_classes
from Plateonlydetect import PlateInferenceConfig
from pathlib import Path
import keras.backend as K

class ModelSingleton:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelSingleton, cls).__new__(cls)
                cls._instance.food_model = None
                cls._instance.plate_model = None
                cls._instance.class_names = None
                cls._instance.food_session = None
                cls._instance.plate_session = None
        return cls._instance
    
    @lru_cache(maxsize=1)
    def load_models(self):
        """Load models once and cache them"""
        ROOT_DIR = Path(__file__).parent.absolute()
        
        # Configure TensorFlow session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Initialize food model
        K.clear_session()
        self.food_session = tf.compat.v1.Session(config=config)
        with self.food_session.as_default():
            with self.food_session.graph.as_default():
                food_config = InferenceConfig()
                self.food_model = modellib.MaskRCNN(
                    mode="inference", 
                    config=food_config, 
                    model_dir=str(ROOT_DIR / "logs")
                )
                self.food_model.load_weights(
                    str(ROOT_DIR / "models/mask_rcnn_food_0300.h5"), 
                    by_name=True
                )
                
        # Initialize plate model
        K.clear_session()
        self.plate_session = tf.compat.v1.Session(config=config)
        with self.plate_session.as_default():
            with self.plate_session.graph.as_default():
                plate_config = PlateInferenceConfig()
                self.plate_model = modellib.MaskRCNN(
                    mode="inference", 
                    config=plate_config, 
                    model_dir=str(ROOT_DIR / "logs")
                )
                self.plate_model.load_weights(
                    str(ROOT_DIR / "models/mask_rcnn_plate_final.h5"), 
                    by_name=True
                )
        
        # Load class names
        meta_path = ROOT_DIR / "food-recognition-dataset/meta.json"
        self.class_names = get_food_classes(str(meta_path))
        
        return self.food_model, self.plate_model, self.class_names

    def detect_food(self, image):
        """Run food detection with proper session handling"""
        with self.food_session.as_default():
            with self.food_session.graph.as_default():
                return self.food_model.detect([image], verbose=0)[0]

    def detect_plate(self, image):
        """Run plate detection with proper session handling"""
        with self.plate_session.as_default():
            with self.plate_session.graph.as_default():
                return self.plate_model.detect([image], verbose=0)[0]
                
    def __del__(self):
        """Cleanup sessions on deletion"""
        if hasattr(self, 'food_session'):
            self.food_session.close()
        if hasattr(self, 'plate_session'):
            self.plate_session.close()