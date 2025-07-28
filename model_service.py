from pathlib import Path
import pickle as pk

from model import build_model
from config import settings
from loguru import logger

class ModelService: 
    def __init__(self):
        logger.info("Initializing ModelService")
        self.model = None
    
    def load_model(self):
        model_path = Path(f"{settings.model_path}/{settings.model_name}")
        logger.info(f"Loading model from {model_path}")

        if not model_path.exists():
            logger.warning("Model file does not exist. Building model...")
            build_model()
        else:
            logger.info("Model file found. Loading existing model.")

        self.model = pk.load(open(f"{settings.model_path}/{settings.model_name}", "rb"))
        logger.info("Model loaded successfully.")

    def predict(self, input_parameters):
        logger.info(f"Making prediction for input: {input_parameters}")
        prediction = self.model.predict([input_parameters])
        logger.info(f"Prediction result: {prediction}")
        return prediction



