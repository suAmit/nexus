import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# We import the same logic the live app uses
from src.prompt_tier_classifier import PromptTierClassifier
from train.dataset import PROMPT_TIER_DATASET


def train_tier_classifier():
    print("Starting Scorer Training...")
    tier_classifier = PromptTierClassifier()

    X = []
    y = []

    for item in PROMPT_TIER_DATASET:
        # Use the logic from Scorer to ensure features match
        features = tier_classifier._extract_features(item["prompt"])
        X.append(features)
        y.append(tier_classifier.inv_label_map[item["tier"]])

    X = np.array(X)
    y = np.array(y)

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Save to the data folder
    os.makedirs("trained_data", exist_ok=True)
    bundle = {
        "classifier": clf,
        "label_map": tier_classifier.label_map,
        "features_count": X.shape[1],
    }

    output_path = "trained_data/tier_classifier_bundle.joblib"
    joblib.dump(bundle, output_path)
    print(f"Success! Model saved to {output_path}")


if __name__ == "__main__":
    train_tier_classifier()
