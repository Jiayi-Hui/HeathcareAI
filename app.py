# -*- coding: utf-8 -*-
"""STAT 8017 - Healthcare Chatbot UI

Single Agent Architecture:
- NB Model Function (Top 5 Prediction)
- RAG Retrieval Function (Evidence & References)
- Decision threshold logic (0.04)
"""

import os

import streamlit as st
from dotenv import load_dotenv

import disease_predictor
from healthcare_agent import HealthcareAgent
from rag_agent import HealthcareRAGAgent

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
def load_rag_agent():
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    return HealthcareRAGAgent(persist_directory=persist_dir)

@st.cache_resource
def load_healthcare_agent():
    rag_agent = load_rag_agent()
    agent = HealthcareAgent(rag_agent=rag_agent, nb_predictor=disease_predictor)
    return agent, rag_agent

if "healthcare_agent" not in st.session_state:
    with st.spinner("Loading Healthcare Agent..."):
        st.session_state.healthcare_agent, st.session_state.rag_agent = load_healthcare_agent()
        stats = st.session_state.rag_agent.get_stats()
        st.success(f"✅ Agent loaded! Knowledge base: {stats['total_chunks']:,} references | Memory: {stats['memory_count']} interactions")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    stats = st.session_state.rag_agent.get_stats()
    st.metric("📊 Knowledge References", f"{stats['total_chunks']:,}")
    st.metric("📝 Interactions", f"{stats['memory_count']}")

    st.divider()
    st.subheader("📁 Upload Healthcare Data")

    csv_path = os.environ.get("CSV_PATH", "")
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
                    result = st.session_state.rag_agent.process_csv(tmp_path)
                    st.success(f"✅ Complete! Rows: {result['rows_processed']:,}, Documents: {result['documents_created']:,}")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.divider()

    if st.button("🗑️ Clear Database", use_container_width=True, type="secondary"):
        if stats['total_chunks'] > 0:
            st.session_state.rag_agent.clear_database()
            st.success("✅ Database cleared!")
            st.rerun()

    if st.button("💬 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_agent.clear_history()
        st.rerun()

    st.divider()
    expand_references = st.checkbox("📚 Expand References by Default", value=False)
    expand_predictions = st.checkbox("🔮 Expand Predictions by Default", value=False)
    expand_symptoms = st.checkbox("🩺 Expand Symptoms by Default", value=False)

# Main content - Two tabs
tab1, tab2 = st.tabs(["💬 Symptom Analysis", "📝 Structured Input"])

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Tab 1: Natural Language Input ===
with tab1:
    st.markdown("Describe your symptoms in natural language or ask a health-related question.")

    # Display existing messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # Check intent - only show symptoms/predictions for SYMPTOM_DESCRIPTION
                intent = message.get("intent", "SYMPTOM_DESCRIPTION")

                if intent == "SYMPTOM_DESCRIPTION":
                    # Show extracted symptoms (always visible, expand state controlled by checkbox)
                    if message.get("symptoms"):
                        with st.expander("🩺 Extracted Symptoms", expanded=expand_symptoms):
                            symptoms = message["symptoms"]
                            symptom_labels = {
                                "fever": "Fever",
                                "cough": "Cough",
                                "fatigue": "Fatigue",
                                "difficulty_breathing": "Difficulty Breathing",
                                "age": "Age Group",
                                "gender": "Gender",
                                "blood_pressure": "Blood Pressure",
                                "cholesterol": "Cholesterol"
                            }
                            for key, label in symptom_labels.items():
                                val = symptoms.get(key, -1)
                                if key in ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]:
                                    display = {"1": "Yes/Present", "0": "No/Absent", "-1": "Unknown"}.get(str(val), "Unknown")
                                    if key == "gender":
                                        display = {"1": "Male", "0": "Female", "-1": "Unknown"}.get(str(val), "Unknown")
                                elif key == "age":
                                    if val == -1:
                                        display = "Unknown"
                                    else:
                                        display = {1: "0-18", 2: "19-35", 3: "36-50", 4: "51-65", 5: "65+"}.get(val, f"{val*10}-{val*10+9} years")
                                elif key == "blood_pressure":
                                    display = {"2": "High", "1": "Normal", "0": "Low", "-1": "Unknown"}.get(str(val), "Unknown")
                                elif key == "cholesterol":
                                    display = {"2": "High", "1": "Normal", "0": "Low", "-1": "Unknown"}.get(str(val), "Unknown")
                                st.write(f"**{label}:** {display}")

                    # Show predictions (always visible, expand state controlled by checkbox)
                    if message.get("predictions"):
                        with st.expander("🔮 Top 5 Predictions", expanded=expand_predictions):
                            is_clear = message.get("is_clear_pattern", False)
                            threshold_status = "✅ Clear Pattern" if is_clear else "⚠️ Uncertain/Complex"
                            st.write(f"**Decision:** {threshold_status} (threshold: 0.04)")

                            predictions = message["predictions"].get("top5", [])
                            for i, (disease, prob) in enumerate(predictions):
                                # Progress bar for probability
                                st.write(f"**{i+1}. {disease}**")
                                st.progress(float(prob))
                                st.write(f"   Probability: {prob:.2%}")

                # Show references (SYMPTOM_DESCRIPTION only)
                if intent != "GENERAL_QUESTION" and message.get("evidence"):
                    with st.expander(f"📖 References ({len(message['evidence'])}) - Symptom Analysis", expanded=expand_references):
                        for i, doc in enumerate(message["evidence"], 1):
                            st.markdown(f"**--- Reference {i} ---**")
                            st.markdown(f"**📌 Title:** {doc['title']}")
                            if 'link' in doc:
                                st.markdown(f"**🔗 Source:** [{doc['link']}]({doc['link']})")
                            st.markdown(f"**📊 Score:** {doc['score']:.4f}")
                            content = doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content", "")
                            st.markdown(f"**📝 Content:** {content}")
                            st.divider()

    # Handle new input
    if prompt := st.chat_input("Describe your symptoms or ask a health question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            intent = st.session_state.healthcare_agent.detect_intent(prompt)
            spinner_msg = "Thinking..." if intent == "GENERAL_QUESTION" else "🔍 Analyzing symptoms with NB Model + RAG Retrieval..."
            with st.spinner(spinner_msg):
                result = st.session_state.healthcare_agent.process_query(prompt)

            st.markdown(result["response"])

            # For SYMPTOM_DESCRIPTION: show quick decision summary
            if intent == "SYMPTOM_DESCRIPTION" and result.get("predictions"):
                is_clear = result.get("is_clear_pattern", False)
                threshold_status = "✅ Clear Pattern" if is_clear else "⚠️ Uncertain/Complex"
                st.info(f"**Decision Logic:** {threshold_status} (max probability: {result['predictions']['max_probability']:.2%}, threshold: 0.04)")

            # Show expanders for current message (default collapsed for new messages)
            if intent == "SYMPTOM_DESCRIPTION":
                # Show symptoms expander
                if result.get("symptoms"):
                    with st.expander("🩺 Extracted Symptoms", expanded=False):
                        symptoms = result["symptoms"]
                        symptom_labels = {
                            "fever": "Fever",
                            "cough": "Cough",
                            "fatigue": "Fatigue",
                            "difficulty_breathing": "Difficulty Breathing",
                            "age": "Age Group",
                            "gender": "Gender",
                            "blood_pressure": "Blood Pressure",
                            "cholesterol": "Cholesterol"
                        }
                        for key, label in symptom_labels.items():
                            val = symptoms.get(key, -1)
                            if key in ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]:
                                display = {"1": "Yes/Present", "0": "No/Absent", "-1": "Unknown"}.get(str(val), "Unknown")
                                if key == "gender":
                                    display = {"1": "Male", "0": "Female", "-1": "Unknown"}.get(str(val), "Unknown")
                            elif key == "age":
                                if val == -1:
                                    display = "Unknown"
                                else:
                                    display = {1: "0-18", 2: "19-35", 3: "36-50", 4: "51-65", 5: "65+"}.get(val, f"{val*10}-{val*10+9} years")
                            elif key == "blood_pressure":
                                display = {"2": "High", "1": "Normal", "0": "Low", "-1": "Unknown"}.get(str(val), "Unknown")
                            elif key == "cholesterol":
                                display = {"2": "High", "1": "Normal", "0": "Low", "-1": "Unknown"}.get(str(val), "Unknown")
                            st.write(f"**{label}:** {display}")

                # Show predictions expander
                if result.get("predictions"):
                    with st.expander("🔮 Top 5 Predictions", expanded=False):
                        predictions = result["predictions"].get("top5", [])
                        for i, (disease, prob) in enumerate(predictions):
                            st.write(f"**{i+1}. {disease}**")
                            st.progress(float(prob))
                            st.write(f"   Probability: {prob:.2%}")

            # Show references expander (SYMPTOM_DESCRIPTION only)
            if intent != "GENERAL_QUESTION" and result.get("evidence"):
                with st.expander(f"📖 References ({len(result['evidence'])}) - Symptom Analysis", expanded=False):
                    for i, doc in enumerate(result["evidence"], 1):
                        st.markdown(f"**--- Reference {i} ---**")
                        st.markdown(f"**📌 Title:** {doc['title']}")
                        if 'link' in doc:
                            st.markdown(f"**🔗 Source:** [{doc['link']}]({doc['link']})")
                        st.markdown(f"**📊 Score:** {doc['score']:.4f}")
                        content = doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content", "")
                        st.markdown(f"**📝 Content:** {content}")
                        st.divider()

                references_to_save = result["evidence"]
            else:
                references_to_save = []

        # Save assistant message with all data
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "intent": result["intent"],
            "symptoms": result["symptoms"],
            "predictions": result["predictions"],
            "evidence": references_to_save,
            "is_clear_pattern": result["is_clear_pattern"]
        })

# === Tab 2: Structured Input ===
with tab2:
    st.markdown("Enter symptoms using structured input for more precise predictions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        fever = st.selectbox("Fever", options=[-1, 0, 1], format_func=lambda x: {"-1": "Unknown", "0": "No", "1": "Yes"}[str(x)])
        cough = st.selectbox("Cough", options=[-1, 0, 1], format_func=lambda x: {"-1": "Unknown", "0": "No", "1": "Yes"}[str(x)])

    with col2:
        fatigue = st.selectbox("Fatigue", options=[-1, 0, 1], format_func=lambda x: {"-1": "Unknown", "0": "No", "1": "Yes"}[str(x)])
        difficulty_breathing = st.selectbox("Difficulty Breathing", options=[-1, 0, 1], format_func=lambda x: {"-1": "Unknown", "0": "No", "1": "Yes"}[str(x)])

    with col3:
        age = st.selectbox("Age Group", options=[-1, 1, 2, 3, 4, 5],
                          format_func=lambda x: {"-1": "Unknown", "1": "0-18", "2": "19-35", "3": "36-50",
                                                 "4": "51-65", "5": "65+"}[str(x)])
        gender = st.selectbox("Gender", options=[-1, 0, 1], format_func=lambda x: {"-1": "Unknown", "0": "Female", "1": "Male"}[str(x)])

    with st.expander("Additional Health Indicators"):
        bp = st.selectbox("Blood Pressure", options=[-1, 0, 1, 2],
                         format_func=lambda x: {"-1": "Unknown", "0": "Low", "1": "Normal", "2": "High"}[str(x)])
        chol = st.selectbox("Cholesterol", options=[-1, 0, 1, 2],
                           format_func=lambda x: {"-1": "Unknown", "0": "Low", "1": "Normal", "2": "High"}[str(x)])

    if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
        # Create symptoms dict
        symptoms = {
            "fever": fever,
            "cough": cough,
            "fatigue": fatigue,
            "difficulty_breathing": difficulty_breathing,
            "age": age,
            "gender": gender,
            "blood_pressure": bp,
            "cholesterol": chol
        }

        with st.spinner(" Running NB Model prediction..."):
            # Direct NB prediction (no parsing needed)
            prediction_result = st.session_state.healthcare_agent.predict_top5(symptoms)

            # Get appropriate evidence
            query_desc = f"Patient with fever={fever}, cough={cough}, fatigue={fatigue}, difficulty_breathing={difficulty_breathing}, age group {age}, gender={gender}, BP={'Low' if bp==0 else 'Normal' if bp==1 else 'High' if bp==2 else 'Unknown'}, Cholesterol={'Low' if chol==0 else 'Normal' if chol==1 else 'High' if chol==2 else 'Unknown'}"
            evidence = st.session_state.healthcare_agent.retrieve_evidence(query_desc, prediction_result)

            # Format response
            response = st.session_state.healthcare_agent.format_response(prediction_result, evidence, query_desc)

        # Display results
        is_clear = prediction_result.get("is_clear_pattern", False)
        threshold_status = "✅ Clear Pattern" if is_clear else "⚠️ Uncertain/Complex"

        st.subheader("📊 Analysis Results")
        st.info(f"**Decision:** {threshold_status} (max probability: {prediction_result['max_probability']:.2%}, threshold: 0.04)")

        # Show predictions
        st.subheader("🔮 Top 5 Predictions")
        predictions = prediction_result.get("top5", [])
        cols = st.columns(5)
        for i, (disease, prob) in enumerate(predictions):
            with cols[i]:
                st.metric(f"#{i+1}", disease, f"{prob:.2%}")

        # Progress bars
        for i, (disease, prob) in enumerate(predictions):
            st.write(f"**{i+1}. {disease}**")
            st.progress(float(prob))

        # Show response
        st.subheader("📝 Response")
        st.markdown(response)

        # Show references
        if evidence:
            st.subheader("📖 References")
            for i, doc in enumerate(evidence, 1):
                with st.expander(f"Reference {i}: {doc['title']}"):
                    st.markdown(f"**📊 Score:** {doc['score']:.4f}")
                    if 'link' in doc:
                        st.markdown(f"**🔗 Source:** [{doc['link']}]({doc['link']})")
                    st.markdown(f"**📝 Content:** {doc.get('content', '')[:500]}...")

        # Save to memory
        st.session_state.rag_agent.save_interaction(
            query=query_desc,
            symptoms=symptoms,
            predictions=predictions,
            response=response
        )
