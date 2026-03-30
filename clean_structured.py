import numpy as np
import pandas as pd
import re

# --- Configuration ---
input_folder = "./Dataset/01. Structured/"
summary_data = []

# =============================================================================
# 1. ai-medical-chatbot.csv
# =============================================================================
dataset_name_1 = "ai-medical-chatbot.csv"
output_name_1 = dataset_name_1.replace(".csv", "_cleaned.csv")
print(f"1. Start processing: {dataset_name_1}...")
df1 = pd.read_csv(input_folder + dataset_name_1)
print(f"Initial rows: {df1.shape[0]}")

df1['Doctor'] = df1['Doctor'].str.replace(r'-->', '', regex=False)
doc_lengths = df1['Doctor'].str.len().fillna(0)
lower_quantile_val = doc_lengths.quantile(0.75)
df1 = df1[doc_lengths >= lower_quantile_val][['Patient', 'Doctor']]

df1 = df1.rename(columns={
    'Patient': 'Patient Question',
    'Doctor': 'Doctor Response'
})

summary_data.append({"No": 1, "Cleaned Dataset": output_name_1, "Final Columns": df1.columns.tolist()})
df1.to_csv(input_folder + output_name_1, index=False)
print("End processing.")
print(f"Final row count: {len(df1)}")
print("-" * 50)

# =============================================================================
# 2. symptom_precaution.csv
# =============================================================================
dataset_name_2 = "symptom_precaution.csv"
output_name_2 = dataset_name_2.replace(".csv", "_cleaned.csv")
print(f"2. Start processing: {dataset_name_2}...")
df2 = pd.read_csv(input_folder + dataset_name_2)
print(f"Initial rows: {df2.shape[0]}")

precaution_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
df2['Precautions'] = df2[precaution_cols].fillna('').agg(lambda x: ', '.join(filter(None, x)), axis=1)
df2 = df2.rename(columns={'Disease': 'Disease Name'})
df2 = df2[['Disease Name', 'Precautions']]

summary_data.append({"No": 2, "Cleaned Dataset": output_name_2, "Final Columns": df2.columns.tolist()})
df2.to_csv(input_folder + output_name_2, index=False)
print("End processing.")
print(f"Final row count: {len(df2)}")
print("-" * 50)

# =============================================================================
# 3. Disease_symptom_and_patient_profile_dataset.csv
# =============================================================================
dataset_name_3 = "Disease_symptom_and_patient_profile_dataset.csv"
output_name_3 = dataset_name_3.replace(".csv", "_cleaned.csv")
print(f"3. Start processing: {dataset_name_3}...")
df3 = pd.read_csv(input_folder + dataset_name_3)
print(f"Initial rows: {df3.shape[0]}")

df3['Fever'] = df3['Fever'].map({'Yes': 'have fever', 'No': 'no fever'})
df3['Cough'] = df3['Cough'].map({'Yes': 'have cough', 'No': 'no cough'})
df3['Fatigue'] = df3['Fatigue'].map({'Yes': 'experience fatigue', 'No': 'no fatigue'})
df3['Difficulty Breathing'] = df3['Difficulty Breathing'].map({'Yes': 'have difficulty breathing', 'No': 'no difficulty breathing'})
df3['Blood Pressure'] = df3['Blood Pressure'].map({'High': 'high blood pressure level', 'Normal': 'normal blood pressure level'})
df3['Cholesterol Level'] = df3['Cholesterol Level'].map({'High': 'high cholesterol level', 'Normal': 'normal cholesterol level'})

def aggregate_medical_status(series):
    positive_keywords = ['have', 'experience', 'high']
    unique_vals = series.dropna().unique()
    for val in unique_vals:
        val_str = str(val)
        if any(word in val_str.lower() for word in positive_keywords):
            return val_str
    return str(unique_vals[0]) if len(unique_vals) > 0 else "unknown status"

df3_grouped = df3.groupby('Disease').agg({
    'Fever': aggregate_medical_status,
    'Cough': aggregate_medical_status,
    'Fatigue': aggregate_medical_status,
    'Difficulty Breathing': aggregate_medical_status,
    'Blood Pressure': aggregate_medical_status,
    'Cholesterol Level': aggregate_medical_status
}).reset_index()

cols_to_concat = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level']
df3_grouped['Symptom Description'] = df3_grouped[cols_to_concat].astype(str).agg(lambda x: ', '.join(x) + '.', axis=1)

df3_final = df3_grouped.rename(columns={'Disease': 'Disease Name'})
df3_final = df3_final[['Disease Name', 'Symptom Description']]

summary_data.append({"No": 3, "Cleaned Dataset": output_name_3, "Final Columns": df3_final.columns.tolist()})
df3_final.to_csv(input_folder + output_name_3, index=False)
print("End processing.")
print(f"Final row count: {len(df3_final)}")
print("-" * 50)

# =============================================================================
# 4. Drug_prescription_to_disease_dataset.csv
# =============================================================================
dataset_name_4 = "Drug_prescription_to_disease_dataset.csv"
output_name_4 = dataset_name_4.replace(".csv", "_cleaned.csv")
print(f"4. Start processing: {dataset_name_4}...")
df4 = pd.read_csv(input_folder + dataset_name_4)
print(f"Initial rows: {df4.shape[0]}")

df4 = df4.drop(df4.columns[0], axis=1)
df4.columns = ['Disease Name', 'Drug Name']
df4['Drug Name'] = df4['Drug Name'].str.replace(r'/', ', ', regex=True)
df4 = df4.groupby('Disease Name')['Drug Name'].apply(lambda x: ', '.join(sorted(set(filter(None, x))))).reset_index()

summary_data.append({"No": 4, "Cleaned Dataset": output_name_4, "Final Columns": df4.columns.tolist()})
df4.to_csv(input_folder + output_name_4, index=False)
print("End processing.")
print(f"Final row count: {len(df4)}")
print("-" * 50)

# =============================================================================
# 5. drugs_side_effects_drugs_com.csv
# =============================================================================
dataset_name_5 = "drugs_side_effects_drugs_com.csv"
output_name_5 = dataset_name_5.replace(".csv", "_cleaned.csv")
print(f"5. Start processing: {dataset_name_5}...")
df5 = pd.read_csv(input_folder + dataset_name_5)
print(f"Initial rows: {df5.shape[0]}")

# Process: drug_name -> Drug Name, medical_condition -> Disease Name, side_effects -> Symptom Description
# and remove all other columns.
df5 = df5.rename(columns={
    'drug_name': 'Drug Name',
    'medical_condition': 'Disease Name',
    'side_effects': 'Symptom Description'
})
df5 = df5[['Drug Name', 'Disease Name', 'Symptom Description']]

summary_data.append({"No": 5, "Cleaned Dataset": output_name_5, "Final Columns": df5.columns.tolist()})
df5.to_csv(input_folder + output_name_5, index=False)
print("End processing.")
print(f"Final row count: {len(df5)}")
print("-" * 50)

# =============================================================================
# 6. Symptom2Disease.csv
# =============================================================================
dataset_name_6 = "Symptom2Disease.csv"
output_name_6 = dataset_name_6.replace(".csv", "_cleaned.csv")
print(f"6. Start processing: {dataset_name_6}...")
df6 = pd.read_csv(input_folder + dataset_name_6)
print(f"Initial rows: {df6.shape[0]}")

df6 = df6.drop(df6.columns[0], axis=1)
df6 = df6.rename(columns={'label': 'Disease Name', 'text': 'Symptom Description'})
df6 = df6.groupby('Disease Name')['Symptom Description'].apply(lambda x: ', '.join(set(filter(None, x)))).reset_index()

summary_data.append({"No": 6, "Cleaned Dataset": output_name_6, "Final Columns": df6.columns.tolist()})
df6.to_csv(input_folder + output_name_6, index=False)
print("End processing.")
print(f"Final row count: {len(df6)}")
print("-" * 50)

# --- Final Summary Table ---
summary_df = pd.DataFrame(summary_data)
print("\n### CLEANED DATASETS SUMMARY ###")
print(summary_df.to_string(index=False))
