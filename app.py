# -*- coding: utf-8 -*-
"""STAT 8017 - Healthcare Chatbot UI"""

import streamlit as st
from rag_agent import HealthcareRAGAgent
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="STAT 8017 Healthcare Chatbot", page_icon="🏥", layout="wide")

st.title("🏥 STAT 8017 Healthcare Chatbot")
st.info("📌 **Project:** STAT 8017 - RAG-based Healthcare Information System (Demonstration)")

st.warning("""
⚠️ **Important Notice:** This chatbot provides general health information from evidence-based sources 
for educational and demonstration purposes only. It is NOT a substitute for professional medical advice, 
diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.
""")

@st.cache_resource
def load_agent():
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    return HealthcareRAGAgent(persist_directory=persist_dir)

if "agent" not in st.session_state:
    with st.spinner("Loading Healthcare RAG Agent..."):
        st.session_state.agent = load_agent()
        stats = st.session_state.agent.get_stats()
        st.success(f"✅ Agent loaded! Database contains {stats['total_chunks']:,} references")

with st.sidebar:
    st.header("⚙️ Settings")
    
    stats = st.session_state.agent.get_stats()
    st.metric("📊 Total References", f"{stats['total_chunks']:,}")
    
    st.divider()
    st.subheader("📁 Upload Healthcare Data")
    
    csv_path = os.environ.get("CSV_PATH", "")
    csv_encoding = os.environ.get("CSV_ENCODING", "auto-detect")
    
    st.info(f"**Data Source:** `{csv_path if csv_path else 'Not set'}`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("📥 Process & Add to Database", use_container_width=True, type="primary"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("Processing... This may take 5-15 minutes"):
                    result = st.session_state.agent.process_csv(tmp_path)
                    st.success(f"✅ Complete! Rows: {result['rows_processed']:,}, Chunks: {result['chunks_created']:,}")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
        if stats['total_chunks'] > 0:
            st.session_state.agent.clear_database()
            st.success("✅ Database cleared!")
            st.rerun()
    
    if st.button("💬 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.clear_history()
        st.rerun()
    
    st.divider()
    show_references = st.checkbox("📚 Show References", value=True)
    max_references = st.slider("Max References", 1, 10, 5)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about healthcare information..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching evidence-based references..."):
            result = st.session_state.agent.chat(prompt)
            st.markdown(result["response"])
    
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    
    if result["retrieved_docs"] and show_references:
        seen = set()
        unique_docs = []
        for doc in result["retrieved_docs"]:
            doc_key = (doc.get('title', ''), doc.get('content', '')[:100])
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)
        
        num_docs = min(len(unique_docs), max_references)
        docs_to_show = unique_docs[:num_docs]
        
        if len(docs_to_show) > 0:
            with st.expander(f"📖 View {num_docs} Reference(s)"):
                for i, doc in enumerate(docs_to_show, 1):
                    st.markdown(f"**--- Reference {i} ---**")
                    st.markdown(f"**📌 Title:** {doc['title']}")
                    if 'link' in doc:
                        st.markdown(f"**🔗 Source:** [{doc['link']}]({doc['link']})")
                    st.markdown(f"**📊 Score:** {doc['score']:.4f}")
                    content = doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content", "")
                    st.markdown(f"**📝 Content:** {content}")
                    st.divider()
