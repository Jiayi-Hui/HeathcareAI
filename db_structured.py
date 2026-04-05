import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# --- 1. Setup & Environment ---
load_dotenv()
db_url = os.getenv("DATABASE_URL")
input_folder = "./Dataset/01. Structured/"

# Mapping filenames to templates based on your specific columns
dataset_templates = {
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
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Create Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS medical_knowledge (
                id SERIAL PRIMARY KEY,
                disease_name TEXT,
                source_file TEXT,
                content TEXT
            );
        """)

        print("Truncating table...")
        cur.execute("TRUNCATE TABLE medical_knowledge RESTART IDENTITY;")
        conn.commit()

        chunk_size = 100

        for file_name, template in dataset_templates.items():
            path = os.path.join(input_folder, file_name)
            if not os.path.exists(path):
                continue

            print(f"Reading: {file_name}")
            df = pd.read_csv(path).fillna("")

            processed_rows = []
            for _, row in df.iterrows():
                # Convert row to dictionary for easier access
                row_dict = row.to_dict()

                # Dynamic extraction of Disease Name (handling potential case sensitivity)
                disease = row_dict.get('Disease Name')

                try:
                    # Format content using the template and the row's data
                    content = template.format(**row_dict)
                    processed_rows.append((disease, file_name, content))
                except KeyError as e:
                    # This happens if a column required by the {} template is missing
                    print(f"  Warning: Missing column {e} in {file_name}")
                    continue

            # Batch Insertion
            for i in range(0, len(processed_rows), chunk_size):
                batch = processed_rows[i : i + chunk_size]
                execute_values(cur, """
                    INSERT INTO medical_knowledge (disease_name, source_file, content)
                    VALUES %s
                """, batch)
                conn.commit()

            print(f"  Inserted {len(processed_rows)} rows.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

def verify_database_upload():
    print("\n" + "="*30)
    print("DATABASE VERIFICATION")
    print("="*30)

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        query = """
            SELECT DISTINCT ON (source_file) 
                   source_file, 
                   disease_name, 
                   content 
            FROM medical_knowledge
            ORDER BY source_file, id;
        """

        cur.execute(query)
        rows = cur.fetchall()

        if not rows:
            print("Verification failed: The table is empty.")
            return

        header = f"{'SOURCE FILE':<55} | {'DISEASE':<25} | {'CONTENT PREVIEW'}"
        print(header)
        print("-" * 120)

        for row in rows:
            # 0=source_file, 1=disease_name, 2=content
            source_file = row[0]
            disease_name = row[1]
            content = row[2].replace('\n', ' ')
            preview = (content[:75] + '...') if len(content) > 75 else content

            print(f"{source_file:<55} | {disease_name:<25} | {preview}")

        print("="*30)
        print(f"Total unique source files verified: {len(rows)}")

    except Exception as e:
        print(f"Verification Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    upload_all_datasets()
    verify_database_upload()
