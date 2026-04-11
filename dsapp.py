# Naive Bayes on dsapp dataset, by Matthew Yuen
# Note: Code below is simple, perform adaptation as needed

# Imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline

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
X_train = np.array(dsapp.drop(["Disease"], axis=1))
y_train = np.array(dsapp["Disease"])
CatNBmodel = Pipeline([("encoder", OrdinalEncoder()), ("catnb", CategoricalNB())])
CatNBmodel.fit(X_train, y_train)
# print(CatNBmodel.score(X_train, y_train))
# Accuracy score = 0.4824

# Check condition of input, return error if incorrect structure, value or out of range
def checkNB(X_pred):
    # X_pred: Features for prediction, list nested with lists/numpy.ndarray input
    # in the order Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender
    # If X_pred is list, converts to numpy.ndarray
    # All variables except age accept [0, 1] (binary variables)
    # Age considers quotient of 10, accept [1, ..., 9]
    # Example: [[0,0,0,0,1,0], [1,1,1,1,9,1]]
    # Note that disease depends on the code previously
    # Checks for X_pred
    if type(X_pred) == list:
        X_pred = np.array(X_pred)
    if type(X_pred) != np.ndarray:
        print("Error: Incorrect X_pred object")
        return X_pred, 0
    if not all(isinstance(i, np.ndarray) for i in X_pred):
        print("Error: Incorrect X_pred object")
        return X_pred, 0
    if [len(x) for x in X_pred] != [6] * len(X_pred):
        print("Error: Incorrect X_pred object")
        return X_pred, 0
    X_pred_binarylist = X_pred[:, [0, 1, 2, 3, 5]].tolist()
    X_pred_agelist = X_pred[:, 4].tolist()
    if not all(set(i).issubset([0, 1]) for i in X_pred_binarylist):
        print("Error: Incorrect X_pred binary/age input(s)")
        return X_pred, 0
    elif not all({i}.issubset([1, 2, 3, 4, 5, 6, 7, 8, 9]) for i in X_pred_agelist):
        print("Error: Incorrect X_pred binary/age input(s)")
        return X_pred, 0
    else:
        print("Correct input")
        # print(X_pred)
        return X_pred, 1

# As cleanNB requires Cleaning and Naive Bayes model training results, can integrate them into the function
def mainNB(X_pred):
    X_pred, checkind = checkNB(X_pred)
    if checkind == 1:
        print("Prediction using CategoricalNB:")
        print(CatNBmodel.predict(X_pred))
        print([diseasev2n[x] for x in list(CatNBmodel.predict(X_pred))])
        print("Prediction probabilities using CategoricalNB:")
        print(CatNBmodel.predict_proba(X_pred))

# Testing
X1 = [[1] * 6]
X2 = np.array(X1)
X3 = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 9, 1]]
X4 = np.array(X3)
X5 = (1, 1, 1, 1, 1, 1)
X6 = [[1] * 5]
X7 = [[1] * 5, [1] * 6]
X8 = [1] * 6
X9 = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 9, 2]]
X10 = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 9, "fish"]]
X11 = [[1] * 6, [0] * 6]
X12 = [[0, 0, 0, 0, 1, 0], [1, 1, 1, 1, "cat", 1]]
mainNB(X1)
mainNB(X2)
mainNB(X3)
mainNB(X4)
mainNB(X5) # Error: Incorrect X_pred object
mainNB(X6) # Error: Incorrect X_pred object
mainNB(X7) # Error: Incorrect X_pred object
mainNB(X8) # Error: Incorrect X_pred object
mainNB(X9) # Error: Incorrect X_pred binary/age input(s)
mainNB(X10) # Error: Incorrect X_pred binary/age input(s)
mainNB(X11) # Error: Incorrect X_pred binary/age input(s)
mainNB(X12) # Error: Incorrect X_pred binary/age input(s)
