# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAT 8017 Group Project - RAG-based Healthcare Information System for educational demonstration. Built by Group 3.1. Uses ZhipuAI (GLM-4.7-flash) as the LLM with HuggingFace embeddings and ChromaDB for vector storage.

## Commands

### Run the Application
```bash
python main.py                  # Launch Streamlit UI (default port 8501)
streamlit run app.py            # Direct Streamlit launch
```

### Environment Setup
```bash
pip install -r requirements.txt  # Install dependencies
```

Required environment variables in `.env`:
- `ZHIPU_API_KEY` - ZhipuAI API key (required)
- `CHROMA_PERSIST_DIR` - ChromaDB storage path (default: `./chroma_db`)
- `STREAMLIT_PORT` - Streamlit port (default: `8501`)

### Data Processing
```bash
python clean_structured.py      # Clean raw datasets in Dataset/01. Structured/
python db_structured.py         # Upload cleaned data to PostgreSQL (requires DATABASE_URL)
```

## Architecture

### Core Components

**`rag_agent.py`** - `HealthcareRAGAgent` class:
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (CPU)
- Vector store: ChromaDB with `healthcare_collection`
- Text splitting: 500 chunk size, 50 overlap
- Methods: `process_csv()`, `retrieve()`, `generate_response()`, `chat()`, `get_stats()`

**`app.py`** - Streamlit UI:
- Uses Streamlit 1.45.0 (note: `st.expander()` does not support `key` parameter in this version)
- Loads agent via `@st.cache_resource`
- Chat interface with unique expander labels (use `#{index}` suffix to avoid duplicate widget labels)
- Message display pattern: render history FIRST, then handle new input
- CSV upload for adding new healthcare data
- Database management controls

**`clean_structured.py`** - Data cleaning:
- Processes 6 raw datasets into standardized `_cleaned.csv` format
- Output columns vary: `Disease Name`, `Symptom Description`, `Drug Name`, `Precautions`, etc.

**`db_structured.py`** - PostgreSQL upload (alternative storage):
- Requires `DATABASE_URL` environment variable
- Creates `medical_knowledge` table with `disease_name`, `source_file`, `content`

**`dsapp.py`** - Naive Bayes classifier experiment:
- Uses `Disease_symptom_and_patient_profile_dataset.csv`
- MultinomialNB and ComplementNB for disease prediction from symptoms
- Note: Model accuracy unsatisfactory (< 0.5), kept for experimentation

### Data Flow

1. Raw CSVs in `Dataset/01. Structured/` → `clean_structured.py` → `_cleaned.csv` files
2. Cleaned CSVs → `rag_agent.process_csv()` → ChromaDB vector store
3. User query → `rag_agent.retrieve()` → similarity search → `rag_agent.generate_response()` → ZhipuAI response

## Dataset Structure

`Dataset/01. Structured/` contains healthcare datasets:
- `ai-medical-chatbot.csv` - Patient/Doctor Q&A pairs
- `Disease_symptom_and_patient_profile_dataset.csv` - Disease-symptom mappings
- `Drug_prescription_to_disease_dataset.csv` - Drug-disease prescriptions
- `drugs_side_effects_drugs_com.csv` - Drug side effects
- `Symptom2Disease.csv` - Symptom-to-disease mapping
- `symptom_precaution.csv` - Disease precautions

Cleaned versions (`*_cleaned.csv`) have standardized column names for processing.

## Important Notes

- This is an **educational demonstration only** - not for medical use
- Always maintain disclaimer in generated responses
- The chatbot must not provide medical advice, diagnosis, or treatment suggestions
- References must be shown with source attribution
- Streamlit widgets with identical labels are treated as duplicates - always use unique labels
- Session state pattern: display existing messages before processing new input to maintain correct order