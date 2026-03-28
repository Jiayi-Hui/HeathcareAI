import json
from pathlib import Path

from unstructured_pipeline.common import iter_unstructured_documents, write_jsonl


INPUT_FOLDER = Path(__file__).resolve().parents[2] / "Dataset" / "02_Unstructured"
OUTPUT_JSONL = INPUT_FOLDER / "unstructured_documents.jsonl"
SUMMARY_JSON = INPUT_FOLDER / "unstructured_documents_summary.json"


def main() -> None:
    counts_by_source = {}
    counts_by_type = {}

    def document_stream():
        for document in iter_unstructured_documents(INPUT_FOLDER):
            source = document["source"]
            source_type = document["source_type"]
            counts_by_source[source] = counts_by_source.get(source, 0) + 1
            counts_by_type[source_type] = counts_by_type.get(source_type, 0) + 1
            yield document

    total = write_jsonl(OUTPUT_JSONL, document_stream())

    summary = {
        "output_file": str(OUTPUT_JSONL),
        "total_documents": total,
        "sources": counts_by_source,
        "source_types": counts_by_type,
        "schema_version": "unstructured_v1",
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {total} documents to {OUTPUT_JSONL}")
    print(f"Wrote summary to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
