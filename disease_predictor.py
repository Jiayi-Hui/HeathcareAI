# Disease prediction module using trained Naive Bayes model
# Uses GaussianNB model trained in dsapp_interaction.py with 13 features

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Model path
MODEL_PATH = os.path.join("Dataset", "01. Structured", "nb_disease_model.pkl")

# Cached model data
_model_data = None


def _load_model(force_reload: bool = False):
    """Load model, optionally forcing a fresh reload."""
    global _model_data
    if _model_data is None or force_reload:
        _model_data = joblib.load(MODEL_PATH)
    return _model_data


def reload_model():
    """Force reload the model from disk (use after retraining)."""
    global _model_data
    _model_data = None
    return _load_model(force_reload=True)


def _compute_interactions(features: dict) -> dict:
    """Compute interaction features from the 4 binary symptoms.

    Uses max(0, val) to treat -1 (unknown) as 0 for interaction calc.
    """
    f = max(0, features.get("Fever", 0))
    c = max(0, features.get("Cough", 0))
    fa = max(0, features.get("Fatigue", 0))
    b = max(0, features.get("Difficulty Breathing", 0))
    return {
        "Fever_and_Fatigue": f * fa,
        "Cough_and_Breathing": c * b,
        "Fever_and_Cough": f * c,
        "All_Four_Symptoms": f * c * fa * b,
        "No_Symptoms": (1 - f) * (1 - c) * (1 - fa) * (1 - b),
    }


def _age_to_bin(age_val: int) -> int:
    """Convert age//10 value to model's age bin.

    Model bins: 1=0-18, 2=19-35, 3=36-50, 4=51-65, 5=65+
    Input age_val: age // 10 (e.g., 2 for age 20-29, 3 for 30-39)
    """
    if age_val == -1:
        return -1
    age_approx = age_val * 10
    if age_approx <= 18:
        return 1
    elif age_approx <= 35:
        return 2
    elif age_approx <= 50:
        return 3
    elif age_approx <= 65:
        return 4
    else:
        return 5


def _prepare_features(
    fever: int, cough: int, fatigue: int, difficulty_breathing: int,
    age: int, gender: int, bp: int = 1, chol: int = 1
) -> pd.DataFrame:
    """Build feature DataFrame with all 13 columns."""
    age_bin = _age_to_bin(age)
    base = {
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue,
        "Difficulty Breathing": difficulty_breathing,
        "Blood Pressure Ord": bp,
        "Cholesterol Ord": chol,
        "Age Bin": age_bin,
        "Gender": gender,
    }
    interactions = _compute_interactions(base)
    base.update(interactions)
    return pd.DataFrame([base])


def get_model_info() -> Dict:
    """Return model metadata including supported diseases."""
    data = _load_model()
    return {
        "num_diseases": len(data["class_names"]),
        "feature_columns": data["feature_cols"],
        "diseases": data["class_names"],
    }


def predict_disease(
    fever: int,
    cough: int,
    fatigue: int,
    difficulty_breathing: int,
    age: int,
    gender: int,
    bp: int = 1,
    chol: int = 1,
) -> Tuple[str, Dict[str, float]]:
    """
    Predict disease from symptoms using trained GaussianNB model.

    Args:
        fever: 0=No, 1=Yes, -1=Unknown
        cough: 0=No, 1=Yes, -1=Unknown
        fatigue: 0=No, 1=Yes, -1=Unknown
        difficulty_breathing: 0=No, 1=Yes, -1=Unknown
        age: Age divided by 10 (e.g., 25 -> 2, 45 -> 4), -1 for unknown
        gender: 0=Female, 1=Male, -1=Unknown
        bp: 0=Low, 1=Normal, 2=High, -1=Unknown (default 1)
        chol: 0=Low, 1=Normal, 2=High, -1=Unknown (default 1)

    Returns:
        Tuple of (predicted_disease_name, probability_dict)
    """
    data = _load_model()
    model = data["model"]
    le = data["label_encoder"]
    class_names = data["class_names"]

    X_pred = _prepare_features(fever, cough, fatigue, difficulty_breathing, age, gender, bp, chol)

    pred_idx = model.predict(X_pred.values)[0]
    pred_name = class_names[pred_idx]
    probs = model.predict_proba(X_pred.values)[0]

    prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probs)}
    return pred_name, prob_dict


def predict_disease_batch(
    symptoms: List[Dict],
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Predict diseases for multiple symptom sets.

    Args:
        symptoms: List of dicts with keys: fever, cough, fatigue,
                  difficulty_breathing, age, gender, bp (optional), chol (optional)

    Returns:
        List of (predicted_disease_name, probability_dict) tuples
    """
    data = _load_model()
    model = data["model"]
    class_names = data["class_names"]

    rows = []
    for s in symptoms:
        rows.append({
            "Fever": s.get("fever", -1),
            "Cough": s.get("cough", -1),
            "Fatigue": s.get("fatigue", -1),
            "Difficulty Breathing": s.get("difficulty_breathing", -1),
            "Blood Pressure Ord": s.get("bp", 1),
            "Cholesterol Ord": s.get("chol", 1),
            "Age Bin": _age_to_bin(s.get("age", -1)),
            "Gender": s.get("gender", -1),
        })

    df = pd.DataFrame(rows)
    interactions = df.apply(
        lambda row: _compute_interactions({
            "Fever": row["Fever"],
            "Cough": row["Cough"],
            "Fatigue": row["Fatigue"],
            "Difficulty Breathing": row["Difficulty Breathing"],
        }),
        axis=1,
        result_type="expand",
    )
    interactions.columns = ["Fever_and_Fatigue", "Cough_and_Breathing", "Fever_and_Cough", "All_Four_Symptoms", "No_Symptoms"]
    df = pd.concat([df, interactions], axis=1)

    predictions = model.predict(df.values)
    probabilities = model.predict_proba(df.values)

    results = []
    for pred, probs in zip(predictions, probabilities):
        disease_name = class_names[pred]
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probs)}
        results.append((disease_name, prob_dict))

    return results


def get_top_diseases(
    fever: int,
    cough: int,
    fatigue: int,
    difficulty_breathing: int,
    age: int,
    gender: int,
    top_n: int = 5,
    bp: int = 1,
    chol: int = 1,
) -> List[Tuple[str, float]]:
    """
    Get top N most likely diseases with their probabilities.
    """
    _, prob_dict = predict_disease(
        fever, cough, fatigue, difficulty_breathing, age, gender, bp, chol
    )

    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:top_n]


def get_top5_with_threshold(
    fever: int,
    cough: int,
    fatigue: int,
    difficulty_breathing: int,
    age: int,
    gender: int,
    threshold: float = 0.04,
    bp: int = 1,
    chol: int = 1,
) -> Dict:
    """
    Get Top 5 predictions with threshold check for decision logic.

    Returns:
        Dict with:
        - top5: List of (disease_name, probability) tuples
        - max_probability: float
        - is_clear_pattern: bool (True if max >= threshold)
        - symptoms: Dict of input symptoms
    """
    top5 = get_top_diseases(
        fever, cough, fatigue, difficulty_breathing, age, gender,
        top_n=5, bp=bp, chol=chol
    )

    max_prob = top5[0][1] if top5 else 0.0
    is_clear_pattern = max_prob >= threshold

    return {
        "top5": top5,
        "max_probability": max_prob,
        "is_clear_pattern": is_clear_pattern,
        "threshold": threshold,
        "symptoms": {
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "difficulty_breathing": difficulty_breathing,
            "age": age,
            "gender": gender,
            "bp": bp,
            "chol": chol,
        }
    }


# Example usage
if __name__ == "__main__":
    info = get_model_info()
    print(f"Model trained on {info['num_diseases']} diseases")
    print(f"Features: {info['feature_columns']}")
    print()

    disease, probs = predict_disease(
        fever=1, cough=1, fatigue=1, difficulty_breathing=0, age=3, gender=1
    )
    print(f"Predicted disease: {disease}")
    print(f"Top probabilities: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print()

    result = get_top5_with_threshold(
        fever=1, cough=1, fatigue=1, difficulty_breathing=0, age=3, gender=1
    )
    print(f"Top 5 predictions (threshold: {result['threshold']}):")
    for disease_name, prob in result["top5"]:
        print(f"  {disease_name}: {prob:.4f}")
    print(f"Max probability: {result['max_probability']:.4f}")
    print(f"Is clear pattern: {result['is_clear_pattern']}")
