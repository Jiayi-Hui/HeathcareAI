# -*- coding: utf-8 -*-
"""STAT 8017 - Healthcare Chatbot UI"""

import streamlit as st
from rag_agent import HealthcareRAGAgent
from evaluator import HealthcareEvaluator
import os
import json
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

@st.cache_resource
def load_evaluator():
    return HealthcareEvaluator()

if "agent" not in st.session_state:
    with st.spinner("Loading Healthcare RAG Agent..."):
        st.session_state.agent = load_agent()
        stats = st.session_state.agent.get_stats()
        st.success(f"✅ Agent loaded! Database contains {stats['total_chunks']:,} references")

if "evaluator" not in st.session_state:
    with st.spinner("Initializing UniEval (T5-based Dialogue Task)..."):
        st.session_state.evaluator = load_evaluator()

if "messages" not in st.session_state:
    st.session_state.messages = []

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

    if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
        st.session_state.agent.clear_database()
        st.success("✅ Database cleared!")
        st.rerun()
    
    st.divider()
    
    show_references = st.checkbox("📚 Show References", value=True)
    max_references = st.slider("Max References to Show", 1, 5, 3)
    
    # Developer session
    st.divider()
    st.header("🛠️ Developer Tools")
    dev_mode = st.toggle("Enable Accuracy Scoreboard", value=False)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
            
            # Developer Mode Accuracy Scoreboard 
            if dev_mode:
                try:
                    # Use the pre-loaded evaluator from session state
                    eval_tool = st.session_state.evaluator
                    
                    ground_truth = None
                    # Standardizing path to test_cases
                    test_path = "eval/test_cases.json"
                    if os.path.exists(test_path):
                        with open(test_path, "r") as f:
                            ground_truth = json.load(f).get(prompt)

                    st.divider()
                    st.subheader("📊 Accuracy Scoreboard")
                    
                    g_score = eval_tool.calculate_grounding_score(result["response"], result["retrieved_docs"])
                    uni = eval_tool.get_unieval_metrics(result["response"])
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Grounding", f"{g_score:.1%}")
                    c2.metric("Coherence", f"{uni.get('coherence', 0):.2f}")
                    
                    if ground_truth:
                        b_score = eval_tool.get_bert_score(result["response"], ground_truth)
                        c3.metric("BERTScore", f"{b_score:.4f}")
                    else:
                        c3.info("No ground truth for this query.")
                except Exception as e:
                    st.error(f"Scoreboard Error: {e}")
    
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    
    if result["retrieved_docs"] and show_references:
        seen = set()
        unique_docs = []
        for doc in result["retrieved_docs"]:
            doc_key = (doc.get('title', ''), doc.get('content', '')[:100])
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)
        
        docs_to_show = unique_docs[:max_references]
        
        if docs_to_show:
            with st.expander(f"📖 View {len(docs_to_show)} Medical Reference(s)"):
                for i, doc in enumerate(docs_to_show, 1):
                    st.markdown(f"**Reference {i}**")
                    if 'title' in doc: st.write(f"📌 **Source:** {doc['title']}")
                    st.info(doc.get("content", "No content available."))
                    if 'score' in doc: st.caption(f"Relevance Score: {doc['score']:.4f}")
                    st.divider()
