# Naive Bayes Disease Prediction Model - Improved Version
# By Matthew Yuen (updated with feature engineering and proper evaluation)

# Imports
import warnings

warnings.filterwarnings("ignore")
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# 1. Data Loading & Cleaning
# ─────────────────────────────────────────────
input_folder = "./Dataset/01. Structured/"
dataset_name = "Disease_symptom_and_patient_profile_dataset.csv"
dsapp = pd.read_csv(input_folder + dataset_name)

# Remove exact duplicate rows
dup_count = dsapp.duplicated().sum()
dsapp = dsapp.drop_duplicates()
print(f"Removed {dup_count} duplicate rows")

# Case normalization
dsapp["Disease"] = dsapp["Disease"].str.lower()
name_variants = {
    "chronic obstructive pulmonary...": "chronic obstructive pulmonary disease (copd)",
    "urinary tract infection (uti)": "urinary tract infection",
}
dsapp["Disease"] = dsapp["Disease"].map(lambda x: name_variants.get(x, x))

# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

# Binary symptoms
dsapp["Fever"] = dsapp["Fever"].map({"Yes": 1, "No": 0})
dsapp["Cough"] = dsapp["Cough"].map({"Yes": 1, "No": 0})
dsapp["Fatigue"] = dsapp["Fatigue"].map({"Yes": 1, "No": 0})
dsapp["Difficulty Breathing"] = dsapp["Difficulty Breathing"].map({"Yes": 1, "No": 0})

# Ordinal features: Blood Pressure and Cholesterol (re-included)
bp_map = {"Low": 0, "Normal": 1, "High": 2}
chol_map = {"Low": 0, "Normal": 1, "High": 2}
dsapp["Blood Pressure Ord"] = dsapp["Blood Pressure"].map(bp_map).fillna(1).astype(int)
dsapp["Cholesterol Ord"] = dsapp["Cholesterol Level"].map(chol_map).fillna(1).astype(int)

# Age: meaningful bins instead of Age // 10
def age_bin(age):
    if age <= 18:
        return 1
    elif age <= 35:
        return 2
    elif age <= 50:
        return 3
    elif age <= 65:
        return 4
    else:
        return 5

dsapp["Age Bin"] = dsapp["Age"].apply(age_bin)

# Gender
dsapp["Gender"] = dsapp["Gender"].map({"Male": 1, "Female": 0})

# Outcome Variable (used for filtering)
dsapp["Outcome Variable"] = dsapp["Outcome Variable"].map({"Positive": 1, "Negative": 0})

# Interaction features — capture symptom co-occurrence
dsapp["Fever_and_Fatigue"] = dsapp["Fever"] * dsapp["Fatigue"]
dsapp["Cough_and_Breathing"] = dsapp["Cough"] * dsapp["Difficulty Breathing"]
dsapp["Fever_and_Cough"] = dsapp["Fever"] * dsapp["Cough"]
dsapp["All_Four_Symptoms"] = (
    dsapp["Fever"] * dsapp["Cough"] * dsapp["Fatigue"] * dsapp["Difficulty Breathing"]
)
dsapp["No_Symptoms"] = (
    (1 - dsapp["Fever"])
    * (1 - dsapp["Cough"])
    * (1 - dsapp["Fatigue"])
    * (1 - dsapp["Difficulty Breathing"])
)

# ─────────────────────────────────────────────
# 3. Data Filtering
# ─────────────────────────────────────────────

# Remove diseases with all negative outcomes
disease_list = dsapp["Disease"].unique().tolist()
for d in disease_list:
    subset = dsapp[dsapp["Disease"] == d]
    if subset["Outcome Variable"].sum() == 0:
        dsapp = dsapp[dsapp["Disease"] != d]

# Remove diseases with fewer than 3 positive samples
disease_list = dsapp["Disease"].unique().tolist()
for d in disease_list:
    subset = dsapp[dsapp["Disease"] == d]
    if (subset["Outcome Variable"] == 1).sum() < 3:
        dsapp = dsapp[dsapp["Disease"] != d]

# Keep only diseases with >= 8 samples for meaningful classification
# Diseases with fewer samples get too lost in the noise
disease_counts = dsapp["Disease"].value_counts()
keep_diseases = disease_counts[disease_counts >= 8].index.tolist()
dsapp = dsapp[dsapp["Disease"].isin(keep_diseases)]

print(f"Final dataset: {len(dsapp)} rows, {dsapp['Disease'].nunique()} classes")
print(f"Classes: {sorted(dsapp['Disease'].unique().tolist())}")

# ─────────────────────────────────────────────
# 4. Prepare Features & Labels
# ─────────────────────────────────────────────

feature_cols = [
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Blood Pressure Ord",
    "Cholesterol Ord",
    "Age Bin",
    "Gender",
    "Fever_and_Fatigue",
    "Cough_and_Breathing",
    "Fever_and_Cough",
    "All_Four_Symptoms",
    "No_Symptoms",
]

X = dsapp[feature_cols].values
le = LabelEncoder()
y = le.fit_transform(dsapp["Disease"])
class_names = le.classes_.tolist()

print(f"\nClass distribution:")
for name, count in zip(class_names, np.bincount(y)):
    print(f"  {name}: {count}")

# ─────────────────────────────────────────────
# 5. Model Comparison with Stratified K-Fold CV
# ─────────────────────────────────────────────

# Use 5-fold if classes have enough samples, otherwise reduce
min_class_count = np.bincount(y).min()
n_splits = min(5, min_class_count)
if n_splits < 2:
    n_splits = 2

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Test multiple alpha values for NB regularization
alpha_candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

models = {
    "MultinomialNB": [MultinomialNB(alpha=a) for a in alpha_candidates],
    "ComplementNB": [ComplementNB(alpha=a) for a in alpha_candidates],
    "GaussianNB": [GaussianNB(var_smoothing=v) for v in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]],
}

print(f"\n{'='*60}")
print(f"Model Comparison — {n_splits}-Fold Stratified Cross-Validation")
print(f"{'='*60}")

best_score = -1
best_model_name = None
best_model = None

for name, model_list in models.items():
    for model in model_list:
        try:
            param_info = ""
            if hasattr(model, "alpha"):
                param_info = f" (alpha={model.alpha})"
            elif hasattr(model, "var_smoothing"):
                param_info = f" (var_smoothing={model.var_smoothing})"

            scoring = {
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
            }
            results = cross_validate(model, X, y, cv=skf, scoring=scoring)

            acc_mean = results["test_accuracy"].mean()
            acc_std = results["test_accuracy"].std()
            f1_mean = results["test_f1_macro"].mean()
            f1_w = results["test_f1_weighted"].mean()

            print(f"  {name}{param_info}:")
            print(f"    Accuracy:  {acc_mean:.4f} (+/- {acc_std:.4f})")
            print(f"    F1 Macro:  {f1_mean:.4f}")
            print(f"    F1 Weighted: {f1_w:.4f}")

            if acc_mean > best_score:
                best_score = acc_mean
                best_model_name = name + param_info
                # Clone the best model for retraining
                if hasattr(model, "alpha"):
                    best_model = type(model)(alpha=model.alpha)
                elif hasattr(model, "var_smoothing"):
                    best_model = type(model)(var_smoothing=model.var_smoothing)

        except Exception as e:
            print(f"  {name}: FAILED — {e}")

# Retrain the best model on all data
print(f"\n{'='*60}")
print(f"Best model: {best_model_name} (CV Accuracy: {best_score:.4f})")
print(f"Retraining on full dataset...")
best_model.fit(X, y)

# ─────────────────────────────────────────────
# 6. Save Model Artifacts for Backend
# ─────────────────────────────────────────────

model_artifacts = {
    "model": best_model,
    "label_encoder": le,
    "feature_cols": feature_cols,
    "class_names": class_names,
}

model_path = os.path.join(input_folder, "nb_disease_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_artifacts, f)

print(f"Model saved to {model_path}")

# ─────────────────────────────────────────────
# 7. Confusion Matrix on Training Data
# ─────────────────────────────────────────────

from sklearn.metrics import confusion_matrix, classification_report

y_pred = best_model.predict(X)
cm = confusion_matrix(y, y_pred)

print(f"\n{'='*60}")
print("Confusion Matrix (training data — for reference only)")
print(f"{'='*60}")

# Print abbreviated confusion matrix
labels = le.classes_
n_classes = len(labels)
# Show per-class: predicted correctly / total
for i, label in enumerate(labels):
    correct = cm[i, i]
    total = cm[i, :].sum()
    pct = correct / total * 100 if total > 0 else 0
    print(f"  {label:40s}: {correct}/{total} ({pct:.0f}%)")

print(f"\nOverall training accuracy: {best_model.score(X, y):.4f}")

# ─────────────────────────────────────────────
# 8. Prediction Interface
# ─────────────────────────────────────────────


def checkNB(X_pred):
    """Validate and convert prediction input.

    X_pred: Features for prediction (list, dict, or DataFrame)
    Feature order: Fever, Cough, Fatigue, Difficulty Breathing,
                   Blood Pressure Ord, Cholesterol Ord, Age Bin, Gender,
                   Fever_and_Fatigue, Cough_and_Breathing, Fever_and_Cough,
                   All_Four_Symptoms, No_Symptoms
    Binary features accept [-1, 0, 1] (-1 = unknown)
    Blood Pressure Ord accepts [-1, 0, 1, 2] (-1 = unknown)
    Cholesterol Ord accepts [-1, 0, 1, 2] (-1 = unknown)
    Age Bin accepts [-1, 1, 2, 3, 4, 5] (-1 = unknown)
    """
    if type(X_pred) == list:
        if len(X_pred) != 13:
            print("Error: X_pred list must have 13 values")
            return X_pred, 0
        keys = feature_cols
        X_pred = dict(zip(keys, [[v] for v in X_pred]))

    if type(X_pred) == dict:
        expected_keys = set(feature_cols)
        if set(X_pred.keys()) != expected_keys:
            print(f"Error: X_pred dict keys mismatch. Expected: {expected_keys}")
            return X_pred, 0
        if not all(isinstance(v, list) for v in X_pred.values()):
            print("Error: X_pred dict values must be lists")
            return X_pred, 0
        if not all(len(v) == len(list(X_pred.values())[0]) for v in X_pred.values()):
            print("Error: X_pred dict value lists must have same length")
            return X_pred, 0
        X_pred = pd.DataFrame(data=X_pred)

    if type(X_pred) != pd.core.frame.DataFrame:
        print("Error: X_pred must be list, dict, or DataFrame")
        return X_pred, 0

    if set(X_pred.columns) != set(feature_cols):
        print(f"Error: X_pred columns mismatch. Expected: {feature_cols}")
        return X_pred, 0

    # Validate binary features
    binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]
    for col in binary_cols:
        vals = set(X_pred[col].unique())
        if not vals.issubset({-1, 0, 1}):
            print(f"Error: {col} must be -1, 0, or 1")
            return X_pred, 0

    # Validate ordinal features
    for col in ["Blood Pressure Ord", "Cholesterol Ord"]:
        vals = set(X_pred[col].unique())
        if not vals.issubset({-1, 0, 1, 2}):
            print(f"Error: {col} must be -1, 0, 1, or 2")
            return X_pred, 0

    # Validate Age Bin
    age_vals = set(X_pred["Age Bin"].unique())
    if not age_vals.issubset({-1, 1, 2, 3, 4, 5}):
        print("Error: Age Bin must be -1, 1, 2, 3, 4, or 5")
        return X_pred, 0

    print("Correct input")
    return X_pred, 1


def auto_fill_features(fever, cough, fatigue, breathing):
    """Auto-compute interaction features from the 4 basic symptoms.

    Accepts -1 for unknown, which maps to 0 for interaction calc.
    Returns all 13 features including BP, Chol, Age, Gender placeholders.
    """
    f = max(0, fever)
    c = max(0, cough)
    fa = max(0, fatigue)
    b = max(0, breathing)
    return {
        "Fever": [fever],
        "Cough": [cough],
        "Fatigue": [fatigue],
        "Difficulty Breathing": [breathing],
        "Fever_and_Fatigue": [f * fa],
        "Cough_and_Breathing": [c * b],
        "Fever_and_Cough": [f * c],
        "All_Four_Symptoms": [f * c * fa * b],
        "No_Symptoms": [(1 - f) * (1 - c) * (1 - fa) * (1 - b)],
    }


def predict_disease(fever, cough, fatigue, breathing,
                    bp=1, chol=1, age_bin=2, gender=0):
    """Simple prediction interface using 4 symptoms + optional params.

    Args:
        fever, cough, fatigue, breathing: 0 or 1 (or -1 for unknown)
        bp: Blood Pressure ordinal 0=Low, 1=Normal, 2=High (default 1)
        chol: Cholesterol ordinal 0=Low, 1=Normal, 2=High (default 1)
        age_bin: 1=0-18, 2=19-35, 3=36-50, 4=51-65, 5=65+ (default 2)
        gender: 0=Female, 1=Male (default 0)

    Returns: predicted disease name and probabilities
    """
    features = auto_fill_features(fever, cough, fatigue, breathing)
    features["Blood Pressure Ord"] = [bp]
    features["Cholesterol Ord"] = [chol]
    features["Age Bin"] = [age_bin]
    features["Gender"] = [gender]

    df_pred = pd.DataFrame(features)
    X_pred = df_pred[feature_cols].values

    pred_idx = best_model.predict(X_pred)[0]
    pred_name = class_names[pred_idx]
    probs = best_model.predict_proba(X_pred)[0]

    # Top 3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], probs[i]) for i in top3_idx]

    return pred_name, top3


# ─────────────────────────────────────────────
# 9. Demo Predictions
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("Demo Predictions")
print(f"{'='*60}")

# Test 1: All symptoms present
name, top3 = predict_disease(fever=1, cough=1, fatigue=1, breathing=1)
print(f"\nAll symptoms present → {name}")
for cls, prob in top3:
    print(f"  {cls:40s}: {prob:.4f}")

# Test 2: Fever + fatigue only
name, top3 = predict_disease(fever=1, cough=0, fatigue=1, breathing=0)
print(f"\nFever + Fatigue only → {name}")
for cls, prob in top3:
    print(f"  {cls:40s}: {prob:.4f}")

# Test 3: Cough + breathing only
name, top3 = predict_disease(fever=0, cough=1, fatigue=0, breathing=1)
print(f"\nCough + Difficulty Breathing only → {name}")
for cls, prob in top3:
    print(f"  {cls:40s}: {prob:.4f}")

# Test 4: No symptoms
name, top3 = predict_disease(fever=0, cough=0, fatigue=0, breathing=0)
print(f"\nNo symptoms → {name}")
for cls, prob in top3:
    print(f"  {cls:40s}: {prob:.4f}")

# Test 5: High BP + High Cholesterol + fever
name, top3 = predict_disease(fever=1, cough=0, fatigue=1, breathing=0, bp=2, chol=2, age_bin=4)
print(f"\nFever + Fatigue + High BP + High Chol + Age 51-65 → {name}")
for cls, prob in top3:
    print(f"  {cls:40s}: {prob:.4f}")
