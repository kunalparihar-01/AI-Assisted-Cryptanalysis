"""
predictor.py — Load the trained ML model and predict cipher types.

This module provides a clean interface between the trained model and the
rest of the application. It handles:
  - Loading the saved model (with fallback: retrain if model file missing)
  - Predicting cipher type from ciphertext
  - Returning probability scores for each class
"""

import os
import sys
import numpy as np
from typing import Optional

# Path to saved model files (relative to project root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "cipher_model.joblib")
METRICS_PATH = os.path.join(_PROJECT_ROOT, "data", "cipher_model_metrics.json")


class CipherPredictor:
    """
    Wraps the trained Random Forest classifier to predict cipher types.

    Usage:
        predictor = CipherPredictor()
        result = predictor.predict("KHOOR ZRUOG")
        # {'predicted': 'Caesar', 'probabilities': {'Caesar': 0.94, ...}}
    """

    def __init__(self):
        self._model = None
        self._label_encoder = None
        self._metrics = None
        self._load_model()

    def _load_model(self):
        """
        Load the trained model from disk.
        If the model file does not exist, train it automatically.
        """
        if not os.path.exists(MODEL_PATH):
            print("[CipherPredictor] Model not found — training now (this may take ~30 seconds)...")
            self._train_and_save()
        self._read_model()

    def _train_and_save(self):
        """Run the training script to generate and save the model."""
        # Import here to avoid circular imports at module load time
        sys.path.insert(0, _PROJECT_ROOT)
        from ml.train import generate_dataset, train_model, save_model, SAMPLES_PER_CLASS
        import os
        data_dir = os.path.join(_PROJECT_ROOT, "data")
        os.makedirs(data_dir, exist_ok=True)
        dataset_path = os.path.join(data_dir, "sample_dataset.csv")
        df = generate_dataset(SAMPLES_PER_CLASS)
        model_data = train_model(df)
        save_model(model_data, MODEL_PATH, df, dataset_path)

    def _read_model(self):
        """Load the model and metrics from disk files."""
        import joblib, json
        data = joblib.load(MODEL_PATH)
        self._model = data["model"]
        self._label_encoder = data["label_encoder"]

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                self._metrics = json.load(f)

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._model is not None

    def predict(self, text: str) -> dict:
        """
        Predict the most likely cipher type for a given ciphertext.

        Args:
            text: The ciphertext string to analyze.

        Returns:
            Dict with:
                - predicted (str): The most likely cipher name
                - probabilities (dict): {cipher_name: probability} for each class
                - confidence (float): Probability of the top prediction
        """
        from ml.features import extract_features

        if not self.is_ready():
            return {"predicted": "Unknown", "probabilities": {}, "confidence": 0.0}

        features = extract_features(text)
        if np.all(features == 0):
            return {"predicted": "Unknown", "probabilities": {}, "confidence": 0.0}

        # Get class probabilities from Random Forest
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        class_names = self._label_encoder.classes_

        # Build probability dict
        prob_dict = {name: float(prob) for name, prob in zip(class_names, proba)}

        # Best prediction
        predicted_idx = int(np.argmax(proba))
        predicted = class_names[predicted_idx]
        confidence = float(proba[predicted_idx])

        return {
            "predicted": predicted,
            "probabilities": prob_dict,
            "confidence": confidence,
        }

    def predict_ranked(self, text: str) -> list[dict]:
        """
        Return all cipher predictions ranked by probability (highest first).

        Returns:
            List of {cipher, probability, percentage} dicts.
        """
        result = self.predict(text)
        if not result["probabilities"]:
            return []

        ranked = sorted(
            result["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {
                "cipher": name,
                "probability": prob,
                "percentage": round(prob * 100, 1),
            }
            for name, prob in ranked
        ]

    @property
    def metrics(self) -> Optional[dict]:
        """Return training/evaluation metrics loaded from disk."""
        return self._metrics

    def get_class_names(self) -> list[str]:
        """Return the list of cipher class names the model was trained on."""
        if self._label_encoder is not None:
            return self._label_encoder.classes_.tolist()
        return []

