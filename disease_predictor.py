# Disease prediction module using trained Naive Bayes models
# Import this module in main.py to use disease prediction functionality

from typing import Dict, List, Tuple, Union

import joblib
import pandas as pd

# Load model on import
_model_data = None

def _load_model(force_reload: bool = False):
    """Load model, optionally forcing a fresh reload."""
    global _model_data
    if _model_data is None or force_reload:
        _model_data = joblib.load("disease_nb_model.pkl")
    return _model_data

def reload_model():
    """Force reload the model from disk (use after retraining)."""
    global _model_data
    _model_data = None
    return _load_model(force_reload=True)

def get_model_info() -> Dict:
    """Return model metadata including accuracy and supported diseases."""
    data = _load_model()
    return {
        "training_accuracy": data["training_accuracy"],
        "num_diseases": len(data["disease_name_to_value"]) - 1,  # exclude "none"
        "feature_columns": data["feature_columns"],
        "diseases": [v for k, v in data["disease_value_to_name"].items() if k != 0]
    }

def predict_disease(
    fever: int,
    cough: int,
    fatigue: int,
    difficulty_breathing: int,
    age: int,
    gender: int,
    model_type: str = "complement"
) -> Tuple[str, Dict[str, float]]:
    """
    Predict disease from symptoms using trained Naive Bayes model.

    Args:
        fever: 0=No, 1=Yes, -1=Unknown
        cough: 0=No, 1=Yes, -1=Unknown
        fatigue: 0=No, 1=Yes, -1=Unknown
        difficulty_breathing: 0=No, 1=Yes, -1=Unknown
        age: Age divided by 10 (e.g., 25 -> 2, 45 -> 4), -1 for unknown
        gender: 0=Female, 1=Male, -1=Unknown
        model_type: "multinomial" or "complement" (default: complement)

    Returns:
        Tuple of (predicted_disease_name, probability_dict)
    """
    data = _load_model()

    # Select model
    if model_type == "multinomial":
        model = data["mulnb_model"]
    else:
        model = data["comnb_model"]

    # Prepare input (shift values by +1 to handle -1 unknowns for MultinomialNB)
    data = _load_model()
    shift = data.get("value_shift", 1)

    X_pred = pd.DataFrame([{
        "Fever": fever + shift,
        "Cough": cough + shift,
        "Fatigue": fatigue + shift,
        "Difficulty Breathing": difficulty_breathing + shift,
        "Age": age + shift,
        "Gender": gender + shift
    }])

    # Predict
    prediction = model.predict(X_pred)[0]
    probabilities = model.predict_proba(X_pred)[0]

    # Map to disease names
    disease_name = data["disease_value_to_name"][prediction]

    # Create probability dict
    prob_dict = {
        data["disease_value_to_name"][i]: float(prob)
        for i, prob in enumerate(probabilities)
    }

    return disease_name, prob_dict

def predict_disease_batch(
    symptoms: Union[List[Dict], pd.DataFrame],
    model_type: str = "complement"
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Predict diseases for multiple symptom sets.

    Args:
        symptoms: List of dicts with keys: fever, cough, fatigue,
                  difficulty_breathing, age, gender
                  OR DataFrame with columns: Fever, Cough, Fatigue,
                  Difficulty Breathing, Age, Gender
        model_type: "multinomial" or "complement" (default: complement)

    Returns:
        List of (predicted_disease_name, probability_dict) tuples
    """
    data = _load_model()

    # Select model
    if model_type == "multinomial":
        model = data["mulnb_model"]
    else:
        model = data["comnb_model"]

    # Prepare input DataFrame
    data = _load_model()
    shift = data.get("value_shift", 1)

    if isinstance(symptoms, list):
        X_pred = pd.DataFrame(symptoms)
        # Rename columns to match training data
        X_pred = X_pred.rename(columns={
            "fever": "Fever",
            "cough": "Cough",
            "fatigue": "Fatigue",
            "difficulty_breathing": "Difficulty Breathing",
            "age": "Age",
            "gender": "Gender"
        })
    else:
        X_pred = symptoms

    # Apply shift to handle -1 unknowns
    X_pred = X_pred + shift

    # Predict
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)

    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        disease_name = data["disease_value_to_name"][pred]
        prob_dict = {
            data["disease_value_to_name"][j]: float(prob)
            for j, prob in enumerate(probs)
        }
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
    model_type: str = "complement"
) -> List[Tuple[str, float]]:
    """
    Get top N most likely diseases with their probabilities.

    Args:
        Same as predict_disease
        top_n: Number of top predictions to return (default: 5)

    Returns:
        List of (disease_name, probability) tuples sorted by probability
    """
    disease_name, prob_dict = predict_disease(
        fever, cough, fatigue, difficulty_breathing, age, gender, model_type
    )

    # Sort by probability (descending) and take top N
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
    model_type: str = "complement"
) -> Dict:
    """
    Get Top 5 predictions with threshold check for decision logic.

    Args:
        Same as predict_disease
        threshold: Probability threshold for clear pattern vs uncertain (default: 0.04)

    Returns:
        Dict with:
        - top5: List of (disease_name, probability) tuples
        - max_probability: float
        - is_clear_pattern: bool (True if max >= threshold)
        - symptoms: Dict of input symptoms
    """
    top5 = get_top_diseases(
        fever, cough, fatigue, difficulty_breathing, age, gender,
        top_n=5, model_type=model_type
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
            "gender": gender
        }
    }


# Example usage
if __name__ == "__main__":
    # Print model info
    info = get_model_info()
    print(f"Model trained on {info['num_diseases']} diseases")
    print(f"Training accuracy: {info['training_accuracy']}")
    print()

    # Example prediction: patient with fever, cough, fatigue, age 30, male
    disease, probs = predict_disease(
        fever=1, cough=1, fatigue=1, difficulty_breathing=0, age=3, gender=1
    )
    print(f"Predicted disease: {disease}")
    print()

    # Get top 5 predictions with threshold check
    result = get_top5_with_threshold(
        fever=1, cough=1, fatigue=1, difficulty_breathing=0, age=3, gender=1
    )
    print(f"Top 5 predictions (threshold: {result['threshold']}):")
    for disease_name, prob in result["top5"]:
        print(f"  {disease_name}: {prob:.4f}")
    print(f"Max probability: {result['max_probability']:.4f}")
    print(f"Is clear pattern: {result['is_clear_pattern']}")
