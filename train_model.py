# Train and export Naive Bayes models for disease prediction
# Simplified version - no Pipeline/OrdinalEncoder (features are already numeric)

import warnings

warnings.filterwarnings("ignore")
import joblib
import pandas as pd
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# Load dataset
input_folder = "./Dataset/01. Structured/"
dataset_name = "Disease_symptom_and_patient_profile_dataset.csv"
dsapp = pd.read_csv(input_folder + dataset_name)

# Cleaning
dsapp = dsapp.drop(dsapp.columns[7:9], axis=1)
dsapp["Disease"] = dsapp["Disease"].str.lower()
dsapp["Fever"] = dsapp["Fever"].map({"Yes": 1, "No": 0})
dsapp["Cough"] = dsapp["Cough"].map({"Yes": 1, "No": 0})
dsapp["Fatigue"] = dsapp["Fatigue"].map({"Yes": 1, "No": 0})
dsapp["Difficulty Breathing"] = dsapp["Difficulty Breathing"].map({"Yes": 1, "No": 0})
dsapp["Age"] = dsapp["Age"] // 10
dsapp["Gender"] = dsapp["Gender"].map({"Male": 1, "Female": 0})
dsapp["Outcome Variable"] = dsapp["Outcome Variable"].map({"Positive": 1, "Negative": 0})
namevar = {"chronic obstructive pulmonary...": "chronic obstructive pulmonary disease (copd)", "urinary tract infection (uti)": "urinary tract infection"}
dsapp["Disease"] = dsapp["Disease"].map(lambda x: namevar.get(x, x))
dsapp = dsapp.sort_values(by="Disease")

disease = dsapp["Disease"].unique().tolist()
dsapprm = pd.DataFrame()
for d in disease:
    dsappd = dsapp[dsapp["Disease"] == d]
    if sum(dsappd["Outcome Variable"]) == 0:
        dsapp = dsapp[dsapp["Disease"] != d]
        dsapprm = pd.concat([dsapprm, dsappd])

disease = dsapp["Disease"].unique().tolist()

for d in disease:
    dsappd = dsapp[dsapp["Disease"] == d]
    dsappd1 = dsappd[dsappd["Outcome Variable"] == 1]
    if len(dsappd1) <= 2:
        dsapp = dsapp[dsapp["Disease"] != d]
        dsapprm = pd.concat([dsapprm, dsappd])

disease = dsapp["Disease"].unique().tolist()

disease_values = range(len(disease)+1)
diseasen2v = dict(zip(["none"] + disease, disease_values))
diseasev2n = dict(zip(disease_values, ["none"] + disease))
dsapp["Disease"] = dsapp["Disease"].map(diseasen2v) * dsapp["Outcome Variable"]
dsapp = dsapp.drop(["Outcome Variable"], axis=1)

# Train models - Direct NB without Pipeline (features are already numeric)
X_train = dsapp.drop(["Disease"], axis=1)
y_train = dsapp["Disease"]

# Handle negative values (-1 for unknown) by shifting all values to be non-negative
# MultinomialNB requires non-negative features
X_train_shifted = X_train + 1  # Shift -1->0, 0->1, 1->2, age 1-9 -> 2-10

MulNBmodel = MultinomialNB()
ComNBmodel = ComplementNB()

MulNBmodel.fit(X_train_shifted, y_train)
ComNBmodel.fit(X_train_shifted, y_train)

# Save models and metadata
model_data = {
    "mulnb_model": MulNBmodel,
    "comnb_model": ComNBmodel,
    "disease_name_to_value": diseasen2v,
    "disease_value_to_name": diseasev2n,
    "feature_columns": list(X_train.columns),
    "value_shift": 1,  # Remember to shift values by +1 before prediction
    "training_accuracy": {
        "multinomial": MulNBmodel.score(X_train_shifted, y_train),
        "complement": ComNBmodel.score(X_train_shifted, y_train)
    }
}

joblib.dump(model_data, "disease_nb_model.pkl")
print("Model saved to disease_nb_model.pkl")
print(f"Training accuracy - MultinomialNB: {model_data['training_accuracy']['multinomial']:.4f}")
print(f"Training accuracy - ComplementNB: {model_data['training_accuracy']['complement']:.4f}")
print(f"Number of diseases: {len(disease)}")
print(f"Feature columns: {model_data['feature_columns']}")
print("Note: Values must be shifted by +1 before prediction (to handle -1 unknowns)")
