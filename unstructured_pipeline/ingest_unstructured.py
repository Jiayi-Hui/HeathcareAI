import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(".env")
load_dotenv(".env.unstructured", override=True)

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = PIPELINE_DIR.parent / "Dataset" / "02_Unstructured" / "unstructured_documents.jsonl"
DEFAULT_CHUNKS_JSONL = PIPELINE_DIR.parent / "Dataset" / "02_Unstructured" / "unstructured_chunks.jsonl"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_zhipu_api_key() -> str:
    for name in ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise RuntimeError("Missing required environment variable: ZHIPUAI_API_KEY or ZHIPU_API_KEY")


def get_text_splitter(chunk_size: int, chunk_overlap: int):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise RuntimeError(
            "LangChain text splitter is unavailable. Install a compatible langchain-text-splitters stack first."
        ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
    )


def get_mongo_collection():
    try:
        import certifi
        from pymongo import MongoClient
    except Exception as exc:
        raise RuntimeError(
            "pymongo and certifi are required for MongoDB storage. Install them before running ingestion."
        ) from exc

    mongo_uri = require_env("MONGODB_URI")
    mongo_db = os.getenv("MONGODB_DB", "healthai")
    mongo_collection = os.getenv("MONGODB_COLLECTION", "unstructured_documents")

    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    return client[mongo_db][mongo_collection]


def get_vectorstore():
    try:
        from langchain_chroma import Chroma
    except Exception as exc:
        raise RuntimeError(
            "langchain-chroma is required for Chroma vector storage."
        ) from exc

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if not embedding_provider:
        embedding_provider = "zhipu" if os.getenv("ZHIPUAI_API_KEY") else "openai"

    if embedding_provider == "zhipu":
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings
        except Exception as exc:
            raise RuntimeError(
                "langchain-community with ZhipuAIEmbeddings support is required for Zhipu embeddings."
            ) from exc
        os.environ["ZHIPUAI_API_KEY"] = get_zhipu_api_key()
        embedding_model = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
        embeddings = ZhipuAIEmbeddings(model=embedding_model)
    elif embedding_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as exc:
            raise RuntimeError("langchain-openai is required for OpenAI embeddings.") from exc
        require_env("OPENAI_API_KEY")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)
    else:
        raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {embedding_provider}")

    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "unstructured_healthcare")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectorstore


def iter_documents(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_chunk_text(document: dict) -> str:
    if document.get("search_text"):
        return document["search_text"]
    return document.get("raw_text", "")


def chunk_document(document: dict, splitter) -> list[dict]:
    base_text = build_chunk_text(document)
    if not base_text:
        return []

    chunks = []
    for chunk_index, chunk_text in enumerate(splitter.split_text(base_text)):
        cleaned = chunk_text.strip()
        if not cleaned:
            continue

        chunk_id = f"{document['doc_id']}_chunk_{chunk_index}"
        metadata = {
            "doc_id": document["doc_id"],
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "source": document["source"],
            "source_file": document["source_file"],
            "source_type": document["source_type"],
            "title": document["title"],
            "url": document["url"],
            "schema_version": document["schema_version"],
        }
        metadata.update(document.get("structured_data", {}))

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document["doc_id"],
                "text": cleaned,
                "metadata": metadata,
            }
        )
    return chunks


def main() -> None:
    input_jsonl = Path(os.getenv("UNSTRUCTURED_DOCS_JSONL", DEFAULT_INPUT_JSONL))
    chunks_jsonl = Path(os.getenv("UNSTRUCTURED_CHUNKS_JSONL", DEFAULT_CHUNKS_JSONL))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    batch_size = int(os.getenv("CHROMA_BATCH_SIZE", "100"))

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mongo_collection = get_mongo_collection()
    vectorstore = get_vectorstore()

    documents = list(iter_documents(input_jsonl))
    if not documents:
        raise RuntimeError(f"No documents found in {input_jsonl}")

    mongo_collection.delete_many({"schema_version": "unstructured_v1"})
    mongo_collection.insert_many(documents)

    all_chunks = []
    with chunks_jsonl.open("w", encoding="utf-8") as f:
        for document in documents:
            chunks = chunk_document(document, splitter)
            all_chunks.extend(chunks)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if not all_chunks:
        raise RuntimeError("Chunking produced no output.")

    from langchain_core.documents import Document

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        vectorstore.add_documents(
            [
                Document(page_content=chunk["text"], metadata=chunk["metadata"])
                for chunk in batch
            ]
        )

    print(f"MongoDB documents inserted: {len(documents)}")
    print(f"Chunk manifest written: {chunks_jsonl}")
    print(f"Chroma chunks inserted: {len(all_chunks)}")


if __name__ == "__main__":
    main()
