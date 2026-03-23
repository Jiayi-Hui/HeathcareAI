import os
import time
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# --- 1. Setup & Environment ---
load_dotenv()
db_url = os.getenv("DATABASE_URL")
input_folder = "./Dataset/01. Structured/"

dataset_templates = {
    "ai-medical-chatbot_cleaned.csv": 
        "Patient asked: {Patient Question}. Symptoms: {Symptom Description}. Doctor Advice: {Doctor Response}",
    "symptom_precaution_cleaned.csv": 
        "For the disease {Disease Name}, the following precautions are advised: {Precautions}",
    "Disease_symptom_and_patient_profile_dataset_cleaned.csv": 
        "A patient diagnosed with {Disease Name} typically exhibits these symptoms: {Symptom Description}",
    "Drug_prescription_to_disease_dataset_cleaned.csv": 
        "Commonly prescribed drugs for {Disease Name} include: {Drug Name}",
    "drugs_side_effects_drugs_com_cleaned.csv": 
        "The medication {Drug Name} is prescribed for {Disease Name}. Side effects/Symptoms: {Symptom Description}",
    "Symptom2Disease_cleaned.csv": 
        "Medical records indicate that {Disease Name} is characterized by: {Symptom Description}"
}

def upload_all_datasets():
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
    except Exception as e:
        print(f"CRITICAL: Connection failed: {e}")
        return

    cur.execute("""
        CREATE TABLE IF NOT EXISTS medical_knowledge (
            id SERIAL PRIMARY KEY,
            disease_name TEXT,
            source_file TEXT,
            content TEXT
        );
    """)
    
    print("Emptying existing data from 'medical_knowledge'...")
    cur.execute("TRUNCATE TABLE medical_knowledge RESTART IDENTITY;")
    conn.commit()

    # Increased to 64 to maximize the 1-request-per-second limit
    chunk_size = 64 

    for file_name, template in dataset_templates.items():
        path = os.path.join(input_folder, file_name)
        if not os.path.exists(path):
            continue
            
        print(f"Processing File: {file_name}")
        df = pd.read_csv(path)
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, chunk_size):
            
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]
            
            texts_to_embed = []
            metadata_list = []
            
            for _, row in chunk_df.iterrows():
                disease = row.get('Disease Name', 'General Consultation')
                try:
                    content = template.format(**row.to_dict())
                    texts_to_embed.append(content)
                    metadata_list.append((disease, file_name, content))
                except KeyError:
                    continue
    cur.close()
    conn.close()
    print("All dataset inserted")

if __name__ == "__main__":
    upload_all_datasets()
