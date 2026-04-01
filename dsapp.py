# Naive Bayes on dsapp dataset, by Matthew Yuen
# Note: Code below is simple, perform adaptation as needed

# Imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import ComplementNB, CategoricalNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
input_folder = "./Dataset/01. Structured/"
# Local testing
# input_folder = "C:/Users/User/Desktop/University courses/STAT8017/Project/Datasets/"
dataset_name = "Disease_symptom_and_patient_profile_dataset.csv"
dsapp = pd.read_csv(input_folder + dataset_name) # dsapp short for Disease_symptom_and_patient_profile
# print(dsapp.head())

# Cleaning
dsapp = dsapp.drop(dsapp.columns[7:9], axis=1) # Removes Blood Pressure and Cholesterol Level
dsapp["Disease"] = dsapp["Disease"].str.lower() # Adjust case or change disease names if needed
dsapp["Fever"] = dsapp["Fever"].map({"Yes": 1, "No": 0})
dsapp["Cough"] = dsapp["Cough"].map({"Yes": 1, "No": 0})
dsapp["Fatigue"] = dsapp["Fatigue"].map({"Yes": 1, "No": 0})
dsapp["Difficulty Breathing"] = dsapp["Difficulty Breathing"].map({"Yes": 1, "No": 0})
dsapp["Age"] = dsapp["Age"] // 10 # Discretizing age by quotient of 10
dsapp["Gender"] = dsapp["Gender"].map({"Male": 1, "Female": 0})
dsapp["Outcome Variable"] = dsapp["Outcome Variable"].map({"Positive": 1, "Negative": 0})
namevar = {"chronic obstructive pulmonary...": "chronic obstructive pulmonary disease (copd)", "urinary tract infection (uti)": "urinary tract infection"}
dsapp["Disease"] = dsapp["Disease"].map(lambda x: namevar.get(x, x)) # Joining diseases with name variants
dsapp = dsapp.sort_values(by="Disease")
# print(dsapp.head())
disease = dsapp["Disease"].unique().tolist()
# print(disease)
dsapp01rm = pd.DataFrame() # For backup purposes
for d in disease:
    dsappd = dsapp[dsapp["Disease"] == d]
    if sum(dsappd["Outcome Variable"]) * sum(1-dsappd["Outcome Variable"]) == 0:
        dsapp = dsapp[dsapp["Disease"] != d] # Remove rows of same disease with same output (all 0s or 1s)
        dsapp01rm = pd.concat([dsapp01rm, dsappd])
print(dsapp.head())
# print(dsapp01rm.head())
disease = dsapp["Disease"].unique().tolist()
# print(disease)

# Naive Bayes
# Model Training
CatNBmodels = [Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ("catnb", MultinomialNB())])
]*len(disease)
ComNBmodels = [Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ("comnb", ComplementNB())])
]*len(disease)
for i in range(len(disease)):
    X_train = pd.DataFrame(dsapp[dsapp["Disease"] == disease[i]]).drop(["Disease", "Outcome Variable"], axis=1)
    y_train = pd.DataFrame(dsapp[dsapp["Disease"] == disease[i]])["Outcome Variable"]
    CatNBmodels[i].fit(X_train, y_train)
    ComNBmodels[i].fit(X_train, y_train)

# For maintenance and checking
# i = 0
# X_pred = {"Fever": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, -1], "Gender": [0, 1]}
# X_pred = pd.DataFrame(data=X_pred)
# print(disease[i])
# print(CatNBmodels[i].predict(X_pred))
# print(ComNBmodels[i].predict(X_pred))
# i = 0
# X_train = pd.DataFrame(dsapp[dsapp["Disease"] == disease[i]]).drop(["Disease", "Outcome Variable"], axis=1)
# y_train = pd.DataFrame(dsapp[dsapp["Disease"] == disease[i]])["Outcome Variable"]
# print(disease[i])
# print(CatNBmodels[i].score(X_train, y_train))
# print(ComNBmodels[i].score(X_train, y_train))

# Model training results are unsatisfactory, i.e. some have accuracy score < 0.5, so worse than random guessing

# Check condition of input, return error if incorrect structure, value or out of range
def checkNB(X_pred, d = "all"):
    # d: Disease name, lowercase text/list input (change above if not lowercase),
    # all if predict for all diseases, else predicts the diseases listed
    # Examples: "all", "allergic rhinitis", ["allergic rhinitis", "alzheimer's disease"]
    # X_pred: Features for prediction, list/dict/pd.core.frame.DataFrame input
    # in the order Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, -1 if not used
    # If X_pred is list, converts to pd.core.frame.DataFrame with 1 row input
    # If X_pred is dict, converts to pd.core.frame.DataFrame
    # All variables except age accept [-1, 0, 1] (binary variables)
    # Age considers quotient of 10, accept [-1, 1, ..., 9]
    # Examples: [0,0,0,0,1,0], [-1,-1,-1,-1,9,-1], {"Fever": [0, 1], ..., "Gender": [0, 1]}
    # Note that disease depends on the code previously
    X = pd.DataFrame(dsapp).drop(["Disease", "Outcome Variable"], axis=1)
    y = pd.DataFrame(dsapp)["Outcome Variable"]
    X_pred_keys = list(X.columns)
    # X_pred = {"Fever": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
    # Checks for d
    if d == "all":
        d = disease
    if type(d) == str:
        d = [d]
    if type(d) != list:
        print("Error: Incorrect disease object")
        return X_pred, d, 0
    if not set(d).issubset(disease):
        print("Error: Incorrect disease name(s)")
        return X_pred, d, 0
    # Checks for X_pred
    if type(X_pred) == list and len(X_pred) != len(X_pred_keys):
        print("Error: Incorrect X_pred list length")
        return X_pred, d, 0
    if type(X_pred) == list:
        X_pred_values = [[i] for i in X_pred]
        X_pred = dict(zip(X_pred_keys, X_pred_values))
    if type(X_pred) == dict and set(X_pred.keys()) != set(X_pred_keys):
        print("Error: Incorrect X_pred dictionary key(s)")
        return X_pred, d, 0
    if type(X_pred) == dict and not all(isinstance(i, list) for i in list(X_pred.values())):
        print("Error: Incorrect X_pred dictionary value type(s)")
        return X_pred, d, 0
    if type(X_pred) == dict and not all(len(i) == len(list(X_pred.values())[0]) for i in list(X_pred.values())):
        print("Error: Incorrect X_pred dictionary value length(s)")
        return X_pred, d, 0
    if type(X_pred) == dict:
        X_pred = pd.DataFrame(data=X_pred)
    if type(X_pred) != pd.core.frame.DataFrame:
        print("Error: Incorrect X_pred object")
        return X_pred, d, 0
    if set(X_pred.columns) != set(X_pred_keys):
        print("Error: Incorrect X_pred column names")
        return X_pred, d, 0
    X_pred_binarylist = X_pred.iloc[:, [0, 1, 2, 3, 5]].values.tolist()
    X_pred_agelist = X_pred.iloc[:, 4].values.tolist()
    if not all(set(i).issubset([-1, 0, 1]) for i in X_pred_binarylist):
        print("Error: Incorrect X_pred binary input(s)")
        return X_pred, d, 0
    elif not all({i}.issubset([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]) for i in X_pred_agelist):
        print("Error: Incorrect X_pred age input(s)")
        return X_pred, d, 0
    else:
        print("Correct input")
        # print(X_pred)
        # print(d)
        return X_pred, d, 1

# As cleanNB requires Cleaning and Naive Bayes model training results, can integrate them into the function
def mainNB(X_pred, d = "all"):
    X_pred, d, checkind = checkNB(X_pred, d)
    if checkind == 1:
        diseaseindex = [disease.index(i) for i in d]
        for i in diseaseindex:
            print("Prediction for " + disease[i] + " using NB:")
            print(CatNBmodels[i].predict(X_pred))
            print("Prediction probabilities:")
            print(CatNBmodels[i].predict_proba(X_pred))
            print("Prediction for " + disease[i] + " using ComplementNB:")
            print(ComNBmodels[i].predict(X_pred))
            print("Prediction probabilities:")
            print(ComNBmodels[i].predict_proba(X_pred))

# Testing
X1 = [1] * 6
X2 = {"Fever": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
X3 = {"Cough": [0, 1], "Fever": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
X4 = pd.DataFrame(X3)
X5 = {"Fever": [0, -1], "Cough": [0, -1], "Fatigue": [0, -1], "Difficulty Breathing": [0, -1], "Age": [-1, 2], "Gender": [0, -1]}
X6 = [1] * 5
X7 = {"Meow": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
X8 = {"Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
X9 = {"Fever": 1, "Cough": 1, "Fatigue": 1, "Difficulty Breathing": 1, "Age": 1, "Gender": 1}
X10 = {"Fever": [1], "Cough": [1], "Fatigue": [1], "Difficulty Breathing": [1], "Age": [1], "Gender": [0, 1]}
X11 = {"Fever": [0, 1], "Cough": [1], "Fatigue": [1], "Difficulty Breathing": [1], "Age": [1], "Gender": [1]}
X12 = (1, 1, 1, 1, 1, 1)
X13 = pd.DataFrame(X7)
X14 = pd.DataFrame(X8)
X15 = {"Fever": [0, 2], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 2], "Gender": [0, 1]}
X16 = {"Fever": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, 0], "Gender": [0, 1]}
d1 = "allergic rhinitis"
d2 = ["allergic rhinitis"]
d3 = ["allergic rhinitis", "alzheimer's disease"]
d4 = ("allergic rhinitis", "alzheimer's disease")
d5 = ["allergic rhinitis", "meow"]
mainNB(X1)
mainNB(X1, d1)
mainNB(X1, d2)
mainNB(X1, d3)
mainNB(X2)
mainNB(X2, d1)
mainNB(X2, d2)
mainNB(X2, d3)
mainNB(X3)
mainNB(X3, d1)
mainNB(X3, d2)
mainNB(X3, d3)
mainNB(X4)
mainNB(X4, d1)
mainNB(X4, d2)
mainNB(X4, d3)
mainNB(X5)
mainNB(X5, d1)
mainNB(X5, d2)
mainNB(X5, d3)
mainNB(X1, d4) # Error: Incorrect disease object
mainNB(X1, d5) # Error: Incorrect disease name(s)
mainNB(X6, d1) # Error: Incorrect X_pred list length
mainNB(X7, d1) # Error: Incorrect X_pred dictionary key(s)
mainNB(X8, d1) # Error: Incorrect X_pred dictionary key(s)
mainNB(X9, d1) # Error: Incorrect X_pred dictionary value type(s)
mainNB(X10, d1) # Error: Incorrect X_pred dictionary value length(s)
mainNB(X11, d1) # Error: Incorrect X_pred dictionary value length(s)
mainNB(X12, d1) # Error: Incorrect X_pred object
mainNB(X13, d1) # Error: Incorrect X_pred column names
mainNB(X14, d1) # Error: Incorrect X_pred column names
mainNB(X15, d1) # Error: Incorrect X_pred binary input(s)
mainNB(X16, d1) # Error: Incorrect X_pred age input(s)
