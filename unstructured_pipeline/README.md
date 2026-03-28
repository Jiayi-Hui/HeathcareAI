# Unstructured Pipeline

Standalone pipeline for collecting, cleaning, and storing unstructured medical text from `WHO`, `NHS`, and `Wikipedia`.

## Structure

- `collection/`
  - source-specific scraping scripts
- `../Dataset/02_Unstructured/`
  - raw files, cleaned CSV files, JSONL documents, and derived outputs
- `common.py`
  - shared schema, mapping rules, and document-building utilities
- `clean_unstructured.py`
  - clean raw unstructured CSV files into standardized CSV outputs
- `clean_unstructured_json.py`
  - convert cleaned source records into JSONL documents with metadata
- `ingest_unstructured.py`
  - chunk documents, store full documents in MongoDB, and store chunk embeddings in ChromaDB

## Output Schema

Standardized fields:

- `Disease Name`
- `Symptom Description`
- `Drug Name`
- `Precautions`
- `Patient Question`
- `Doctor Response`

Document-level metadata includes:

- `doc_id`
- `source`
- `source_file`
- `source_type`
- `title`
- `url`
- `row_index`
- `raw_text`
- `search_text`

## Typical Workflow

1. Run scraping scripts in `collection/` if raw source files need to be regenerated.
2. Run:

```bash
python3 unstructured_pipeline/cleaning/clean_unstructured.py
```

3. Generate JSONL documents with metadata:

```bash
python3 unstructured_pipeline/cleaning/clean_unstructured_json.py
```

4. Configure MongoDB / embedding / Chroma settings in `.env.unstructured`.
5. Run ingestion:

```bash
python3 unstructured_pipeline/ingest_unstructured.py
```

## Notes

- This pipeline is isolated from the main app flow and does not require modifying `db_structured.py` or `rag_agent.py`.
- Current ingestion is configured to use `Zhipu` embeddings by default, with optional `OpenAI` support.
