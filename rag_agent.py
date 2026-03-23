# -*- coding: utf-8 -*-
"""STAT 8017 - Healthcare RAG Agent"""

import os
import pandas as pd
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from zhipuai import ZhipuAI
import hashlib
from tqdm import tqdm
import chardet

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

class HealthcareRAGAgent:
    def __init__(self, persist_directory: str = "./chroma_db"):
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not found. Please set it in .env file")
        
        self.client = ZhipuAI(api_key=api_key)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[".", "!", "?", ";", ",", ""]
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="healthcare_collection"
        )
        
        self.chat_history = []
        print("✅ Healthcare RAG Agent initialized!")

    def _detect_encoding(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            raw_data = b''.join([f.readline() for _ in range(100)])
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        columns = df.columns.tolist()
        col_mapping = {}
        
        title_keywords = ['title', 'name', 'article', 'topic', 'subject', 'heading', 'page']
        content_keywords = ['content', 'text', 'body', 'article', 'description', 'summary', 'wiki', 'extract']
        link_keywords = ['link', 'url', 'source', 'href', 'website', 'web']
        
        used_columns = set()
        
        for col in columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in title_keywords) and col not in used_columns:
                col_mapping['title'] = col
                used_columns.add(col)
                break
        
        for col in columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in content_keywords) and col not in used_columns:
                col_mapping['content'] = col
                used_columns.add(col)
                break
        
        for col in columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in link_keywords) and col not in used_columns:
                col_mapping['link'] = col
                used_columns.add(col)
                break
        
        remaining_cols = [col for col in columns if col not in used_columns]
        col_mapping['other_columns'] = remaining_cols
        col_mapping['all_columns'] = columns
        
        return col_mapping

    def process_csv(self, csv_path: str, encoding: str = None) -> Dict[str, Any]:
        if encoding is None:
            encoding = self._detect_encoding(csv_path)
            print(f"🔍 Auto-detected encoding: {encoding}")
        
        encodings_to_try = [encoding, 'utf-8', 'big5', 'gbk', 'gb18030', 'latin-1']
        encodings_to_try = list(dict.fromkeys(encodings_to_try))
        
        df = None
        successful_encoding = None
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                successful_encoding = enc
                print(f"✅ Successfully read with encoding: {enc}")
                break
            except Exception as e:
                print(f"⚠️ Failed with {enc}: {str(e)[:50]}")
                continue
        
        if df is None:
            raise Exception("Could not read CSV with any encoding!")
        
        print(f"📊 Loaded CSV with {len(df):,} rows")
        col_mapping = self._detect_columns(df)
        print(f"📊 Column mapping: {col_mapping}")
        
        documents = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                title_col = col_mapping.get('title', list(df.columns)[0])
                title = str(row.get(title_col, f'Doc_{idx}'))
                
                content_parts = []
                if title and title != 'nan':
                    content_parts.append(f"Title: {title}")
                
                content_col = col_mapping.get('content', list(df.columns)[-1])
                content = str(row.get(content_col, ''))
                if content and content != 'nan':
                    content_parts.append(f"Content: {content}")
                
                link_col = col_mapping.get('link')
                if link_col:
                    link = str(row.get(link_col, ''))
                    if link and link != 'nan':
                        content_parts.append(f"Source Link: {link}")
                
                for col in col_mapping.get('other_columns', []):
                    value = str(row.get(col, ''))
                    if value and value != 'nan':
                        content_parts.append(f"{col}: {value}")
                
                full_content = "\n".join(content_parts)
                doc_id = hashlib.md5(f"{title}_{idx}".encode('utf-8', errors='ignore')).hexdigest()
                chunks = self.text_splitter.split_text(full_content)
                
                metadata = {
                    "title": title.encode('utf-8', errors='ignore').decode('utf-8'),
                    "source": "healthcare",
                    "doc_id": doc_id,
                    "row_index": int(idx)
                }
                
                if link_col:
                    link = str(row.get(link_col, ''))
                    if link and link != 'nan':
                        metadata["link"] = link
                
                for chunk in chunks:
                    if len(chunk.strip()) > 50:
                        documents.append(Document(page_content=chunk, metadata=metadata.copy()))
                        
            except Exception as e:
                print(f"⚠️ Skipped row {idx}: {str(e)[:50]}")
                continue
        
        print(f"\n📦 Adding {len(documents):,} chunks to ChromaDB...")
        self.vectorstore.add_documents(documents)
        
        return {
            "rows_processed": len(df),
            "chunks_created": len(documents),
            "status": "success",
            "encoding_used": successful_encoding
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        retrieved = []
        for doc, score in results:
            doc_info = {
                "content": doc.page_content,
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score)
            }
            # ✅ FIXED: Was "doc.meta" - should be "doc.metadata" with colon
            if "link" in doc.metadata:
                doc_info["link"] = doc.metadata["link"]
            retrieved.append(doc_info)
        return retrieved

    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return "❌ No relevant information found. Please try a different query."
        
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"Reference {i}:\n"
            context += f"  Title: {doc['title']}\n"
            if 'link' in doc:
                context += f"  Link: {doc['link']}\n"
            context += f"  Content: {doc['content']}\n\n"
        
        system_prompt = f"""You are a healthcare information assistant created by Group 3.1 for STAT 8017 project demonstration.

IMPORTANT GUIDELINES:
1. All information MUST be based ONLY on the provided references from official, evidence-based sources.
2. DO NOT provide medical advice, diagnosis, or treatment suggestions.
3. DO NOT recommend specific medications, dosages, or treatment plans.
4. ALWAYS include a disclaimer that this is for educational/demonstration purposes only.
5. ALWAYS advise users to consult qualified healthcare professionals for medical concerns.
6. If information is not available in the references, clearly state that you cannot answer.
7. Present information objectively without making recommendations.
8. Remember you are created by Group 3.1 for the STAT 8017 project.

References:
{context}"""
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}"}
                ],
                temperature=0.7,
                max_tokens=10240
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API error: {str(e)}"

    def chat(self, query: str) -> Dict[str, Any]:
        retrieved_docs = self.retrieve(query)
        response = self.generate_response(query, retrieved_docs)
        self.chat_history.append({"query": query, "response": response})
        
        return {
            "query": query,
            "response": response,
            "retrieved_docs": retrieved_docs,
            "history_length": len(self.chat_history)
        }

    def clear_database(self):
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="healthcare_collection"
        )
        self.chat_history = []
        return {"status": "cleared"}

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.vectorstore._collection.count()
            return {"total_chunks": count, "status": "success"}
        except:
            return {"total_chunks": 0, "status": "empty"}

    def clear_history(self):
        self.chat_history = []
