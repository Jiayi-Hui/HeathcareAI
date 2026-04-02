# Naive Bayes on dsapp dataset, by Matthew Yuen
# Note: Code below is simple, perform adaptation as needed

# Imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
# input_folder = "./Dataset/01. Structured/"
# Local testing
input_folder = "C:/Users/User/Desktop/University courses/STAT8017/Project/Datasets/"
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
dsapprm = pd.DataFrame() # For backup purposes
for d in disease:
    dsappd = dsapp[dsapp["Disease"] == d]
    if sum(dsappd["Outcome Variable"]) == 0:
        dsapp = dsapp[dsapp["Disease"] != d] # Remove rows of same disease with all 0s
        dsapprm = pd.concat([dsapprm, dsappd])
# print(dsapp.head())
# print(dsapprm)
disease = dsapp["Disease"].unique().tolist()
# print(disease)

for d in disease:
    dsappd = dsapp[dsapp["Disease"] == d]
    dsappd1 = dsappd[dsappd["Outcome Variable"] == 1]
    if len(dsappd1) <= 2:
        dsapp = dsapp[dsapp["Disease"] != d] # Remove rows with < 3 positives given a disease
        dsapprm = pd.concat([dsapprm, dsappd])
# print(dsapp.head())
# print(dsapprm.head())
disease = dsapp["Disease"].unique().tolist()
# print(len(disease))

# for d in disease:
#     print(d)
#     print(len(dsapp[dsapp["Disease"] == d]))
#     print(len(dsapp[dsapp["Disease"] == d][dsapp["Outcome Variable"] == 1]))

disease_values = range(len(disease)+1)
diseasen2v = dict(zip(["none"] + disease, disease_values)) # For conversion from disease names to disease values
diseasev2n = dict(zip(disease_values, ["none"] + disease)) # For conversion from disease values to disease names
dsapp["Disease"] = dsapp["Disease"].map(diseasen2v) * dsapp["Outcome Variable"]
dsapp = dsapp.drop(["Outcome Variable"], axis=1)

# Naive Bayes
# Model Training
MulNBmodel = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ("mulnb", MultinomialNB())])
ComNBmodel = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ("comnb", ComplementNB())])
X_train = dsapp.drop(["Disease"], axis=1)
y_train = dsapp["Disease"]
MulNBmodel.fit(X_train, y_train)
ComNBmodel.fit(X_train, y_train)

# For maintenance and checking
# X_pred = {"Fever": [0, 1], "Cough": [0, 1], "Fatigue": [0, 1], "Difficulty Breathing": [0, 1], "Age": [1, -1], "Gender": [0, 1]}
# X_pred = pd.DataFrame(data=X_pred)
# print(MulNBmodel.predict(X_pred), MulNBmodel.predict_proba(X_pred))
# print(ComNBmodel.predict(X_pred), ComNBmodel.predict_proba(X_pred))
# print(MulNBmodel.score(X_train, y_train))
# print(ComNBmodel.score(X_train, y_train))

# Model training results are unsatisfactory, i.e. accuracy scores < 0.5

# Check condition of input, return error if incorrect structure, value or out of range
def checkNB(X_pred):
    # X_pred: Features for prediction, list/dict/pd.core.frame.DataFrame input
    # in the order Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, -1 if not used / unknown
    # If X_pred is list, converts to pd.core.frame.DataFrame with 1 row input
    # If X_pred is dict, converts to pd.core.frame.DataFrame
    # All variables except age accept [-1, 0, 1] (binary variables)
    # Age considers quotient of 10, accept [-1, 1, ..., 9]
    # Examples: [0,0,0,0,1,0], [-1,-1,-1,-1,9,-1], {"Fever": [0, 1], ..., "Gender": [0, 1]}
    # Note that disease depends on the code previously
    X_pred_keys = list(X_train.columns)
    # Checks for X_pred
    if type(X_pred) == list and len(X_pred) != len(X_pred_keys):
        print("Error: Incorrect X_pred list length")
        return X_pred, 0
    if type(X_pred) == list:
        X_pred_values = [[i] for i in X_pred]
        X_pred = dict(zip(X_pred_keys, X_pred_values))
    if type(X_pred) == dict and set(X_pred.keys()) != set(X_pred_keys):
        print("Error: Incorrect X_pred dictionary key(s)")
        return X_pred, 0
    if type(X_pred) == dict and not all(isinstance(i, list) for i in list(X_pred.values())):
        print("Error: Incorrect X_pred dictionary value type(s)")
        return X_pred, 0
    if type(X_pred) == dict and not all(len(i) == len(list(X_pred.values())[0]) for i in list(X_pred.values())):
        print("Error: Incorrect X_pred dictionary value length(s)")
        return X_pred, 0
    if type(X_pred) == dict:
        X_pred = pd.DataFrame(data=X_pred)
    if type(X_pred) != pd.core.frame.DataFrame:
        print("Error: Incorrect X_pred object")
        return X_pred, 0
    if set(X_pred.columns) != set(X_pred_keys):
        print("Error: Incorrect X_pred column names")
        return X_pred, 0
    X_pred_binarylist = X_pred.iloc[:, [0, 1, 2, 3, 5]].values.tolist()
    X_pred_agelist = X_pred.iloc[:, 4].values.tolist()
    if not all(set(i).issubset([-1, 0, 1]) for i in X_pred_binarylist):
        print("Error: Incorrect X_pred binary input(s)")
        return X_pred, 0
    elif not all({i}.issubset([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]) for i in X_pred_agelist):
        print("Error: Incorrect X_pred age input(s)")
        return X_pred, 0
    else:
        print("Correct input")
        # print(X_pred)
        return X_pred, 1

# As cleanNB requires Cleaning and Naive Bayes model training results, can integrate them into the function
def mainNB(X_pred):
    X_pred, checkind = checkNB(X_pred)
    if checkind == 1:
        print("Prediction using MultinomialNB:")
        print(MulNBmodel.predict(X_pred))
        print(pd.DataFrame(MulNBmodel.predict(X_pred)).iloc[:, 0].map(diseasev2n))
        print("Prediction probabilities using MultinomialNB:")
        print(MulNBmodel.predict_proba(X_pred))
        print("Prediction using ComplementNB:")
        print(ComNBmodel.predict(X_pred))
        print(pd.DataFrame(ComNBmodel.predict(X_pred)).iloc[:, 0].map(diseasev2n))
        print("Prediction probabilities using ComplementNB:")
        print(ComNBmodel.predict_proba(X_pred))

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
mainNB(X1)
mainNB(X2)
mainNB(X3)
mainNB(X4)
mainNB(X5)
mainNB(X6) # Error: Incorrect X_pred list length
mainNB(X7) # Error: Incorrect X_pred dictionary key(s)
mainNB(X8) # Error: Incorrect X_pred dictionary key(s)
mainNB(X9) # Error: Incorrect X_pred dictionary value type(s)
mainNB(X10) # Error: Incorrect X_pred dictionary value length(s)
mainNB(X11) # Error: Incorrect X_pred dictionary value length(s)
mainNB(X12) # Error: Incorrect X_pred object
mainNB(X13) # Error: Incorrect X_pred column names
mainNB(X14) # Error: Incorrect X_pred column names
mainNB(X15) # Error: Incorrect X_pred binary input(s)
mainNB(X16) # Error: Incorrect X_pred age input(s)
