import logging
import os

log_dir = "monitoring/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def log_prediction(input_data, prediction):
    logging.info(f"Input: {input_data}, Prediction: {prediction}")
