"""Train a gesture classifier from a CSV produced by capture_gestures.py.

Usage:
    python train_gesture_model.py --data data/gestures_keyboard.csv
    python train_gesture_model.py --data data/gestures_keyboard.csv --model mlp
"""

import argparse
import csv
import os
import re

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import config
import gesture_features


def load_dataset(csv_path):
    features, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            label = row[0]
            raw = np.array(row[2:], dtype=np.float64).reshape(21, 3)
            features.append(gesture_features.extract_features(raw))
            labels.append(label)
    return np.array(features), np.array(labels)


def default_out_path(data_path):
    name = os.path.basename(data_path)
    match = re.match(r"gestures_(.+)\.csv$", name)
    role = match.group(1) if match else "model"
    return os.path.join(config.MODELS_DIR, f"gesture_classifier_{role}.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf")
    args = parser.parse_args()

    out_path = args.out or default_out_path(args.data)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    X, y = load_dataset(args.data)
    print(f"Loaded {len(y)} samples across {len(set(y))} labels: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if args.model == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification report (held-out test split):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(clf.classes_)
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

    joblib.dump({"model": clf, "labels": clf.classes_.tolist()}, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
