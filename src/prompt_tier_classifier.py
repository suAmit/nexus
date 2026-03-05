import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


class PromptTierClassifier:
    def __init__(self):
        # 1. Setup paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(
            self.base_dir, "trained_data", "tier_classifier_bundle.joblib"
        )

        # 2. Initialize Embedder (Shared between training and inference)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.label_map = {0: "LITE", 1: "MID", 2: "PRO"}

        # Mapping for training script to use
        self.inv_label_map = {"LITE": 0, "MID": 1, "PRO": 2}

        # 3. Heuristic Indicators
        self.pro_keywords = [
            "architect",
            "optimize",
            "refactor",
            "analyze",
            "complex",
            "security",
            "diagram",
            "simulate",
        ]

        # 4. Load the Model
        if os.path.exists(self.model_path):
            self.load_bundle()
        else:
            print("No model bundle found! Run 'python -m train.trainer' first.")
            self.clf = None

    def _extract_features(self, prompt):
        """
        Calculates the 388 features (384 embeddings + 4 metadata).
        This method MUST be used by the trainer to avoid feature mismatch.
        """
        # A. Semantic Embedding (384 features)
        embedding = self.embedder.encode([prompt])[0]

        # B. Structural Features (4 features)
        length = len(prompt)
        word_count = len(prompt.split())
        has_code_request = 1 if "```" in prompt or "code" in prompt.lower() else 0
        has_pro_keyword = (
            1 if any(k in prompt.lower() for k in self.pro_keywords) else 0
        )

        # Normalization (keeps values roughly between 0 and 1)
        meta_features = np.array(
            [length / 500, word_count / 100, has_code_request, has_pro_keyword]
        )

        return np.hstack([embedding, meta_features])

    def predict_tier(self, prompt):
        """Predicts the best tier using hybrid intelligence."""
        # --- Heuristic Overrides ---
        if len(prompt.split()) < 4:
            return "LITE", 1.0

        if any(k in prompt.lower() for k in self.pro_keywords) and len(prompt) > 100:
            return "PRO", 0.95

        # --- ML Prediction ---
        if self.clf is None:
            return "MID", 0.5  # Safety Fallback if no model is found

        features = self._extract_features(prompt).reshape(1, -1)
        probs = self.clf.predict_proba(features)[0]

        detail = f"L: {probs[0]:.1%}, M: {probs[1]:.1%}, P: {probs[2]:.1%}"

        tier = self.label_map[np.argmax(probs)]
        conf = np.max(probs)

        return tier, conf, detail

    def load_bundle(self):
        """Loads the classifier from the data folder."""
        try:
            bundle = joblib.load(self.model_path)
            self.clf = bundle["classifier"]
            # Ensure the loaded model matches our internal maps
            self.label_map = bundle.get("label_map", self.label_map)
        except Exception as e:
            print(f"Error loading bundle: {e}")
            self.clf = None
