# -*- coding: utf-8 -*-
"""STAT 8017 - Healthcare Agent Orchestrator

Single Agent pattern with:
- NB Model Function (Top 5 Prediction)
- RAG Retrieval Function (Evidence & References)
- Decision threshold logic (0.04)
"""

import os
import warnings
from typing import Any, Dict, List

from dotenv import load_dotenv
from zhipuai import ZhipuAI

warnings.filterwarnings("ignore")

load_dotenv()


class HealthcareAgent:
    """Single Agent orchestrating NB Model + RAG Retrieval functions."""

    THRESHOLD = 0.04  # Probability threshold for clear pattern vs uncertain

    def __init__(self, rag_agent, nb_predictor):
        """
        Initialize Healthcare Agent.

        Args:
            rag_agent: HealthcareRAGAgent instance for RAG retrieval
            nb_predictor: disease_predictor module for NB predictions
        """
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not found. Please set it in .env file")

        self.client = ZhipuAI(api_key=api_key)
        self.rag_agent = rag_agent
        self.nb_predictor = nb_predictor
        print("✅ Healthcare Agent initialized!")

    def detect_intent(self, user_input: str) -> str:
        """
        Detect if user input is symptom description or general question.

        Returns:
            "SYMPTOM_DESCRIPTION" - user describing symptoms, should use NB + RAG
            "GENERAL_QUESTION" - user asking about health topic, should use pure RAG
        """
        text = user_input.lower()

        # Check for symptom keywords
        symptom_keywords = [
            "fever", "cough", "fatigue", "tired", "difficulty breathing",
            "shortness of breath", "pain", "headache", "nausea", "vomiting",
            "dizziness", "chest pain", "stomach", "sore throat", "runny nose",
            "i have", "i feel", "my symptoms", "suffering from", "experiencing",
            "hurt", "ache", "swelling", "rash", "itch", "numbness"
        ]

        # Check for question patterns
        question_patterns = [
            "what is", "what are", "how does", "how to", "why",
            "explain", "define", "tell me about", "information about",
            "what's", "whats"
        ]

        has_symptoms = any(kw in text for kw in symptom_keywords)
        has_question = any(qp in text for qp in question_patterns)

        if has_symptoms and not has_question:
            return "SYMPTOM_DESCRIPTION"
        elif has_question and not has_symptoms:
            return "GENERAL_QUESTION"
        elif has_symptoms and has_question:
            # Mixed case - prefer symptom analysis if symptoms are explicit
            return "SYMPTOM_DESCRIPTION"
        else:
            # Ambiguous - default to general RAG
            return "GENERAL_QUESTION"

    def parse_symptoms(self, user_input: str) -> Dict:
        """
        Extract structured symptoms from natural language using ZhipuAI.
        Falls back to keyword-based extraction if LLM fails.

        Args:
            user_input: Natural language description of symptoms

        Returns:
            Dict with fever, cough, fatigue, difficulty_breathing, age, gender
            Values: 1=Yes, 0=No, -1=Unknown
        """
        # First try keyword-based extraction (reliable fallback)
        symptoms = self._extract_symptoms_keywords(user_input)

        # Then try LLM extraction for more nuance
        try:
            llm_symptoms = self._extract_symptoms_llm(user_input)
            # Use LLM results for any values that are still unknown
            for key in symptoms:
                if symptoms[key] == -1 and llm_symptoms.get(key, -1) != -1:
                    symptoms[key] = llm_symptoms[key]
        except Exception as e:
            print(f"⚠️ LLM symptom parsing failed, using keyword extraction: {e}")

        return symptoms

    def _extract_symptoms_keywords(self, user_input: str) -> Dict:
        """Extract symptoms using keyword matching (reliable fallback)."""
        text = user_input.lower()

        symptoms = {
            "fever": -1,
            "cough": -1,
            "fatigue": -1,
            "difficulty_breathing": -1,
            "age": -1,
            "gender": -1
        }

        # Fever detection
        if any(kw in text for kw in ["fever", "high temperature", "feeling hot", "temperature"]):
            symptoms["fever"] = 1
        elif "no fever" in text:
            symptoms["fever"] = 0

        # Cough detection
        if any(kw in text for kw in ["cough", "coughing"]):
            symptoms["cough"] = 1
        elif "no cough" in text:
            symptoms["cough"] = 0

        # Fatigue detection
        if any(kw in text for kw in ["fatigue", "tired", "exhausted", "weakness", "low energy", "feeling tired"]):
            symptoms["fatigue"] = 1
        elif "no fatigue" in text:
            symptoms["fatigue"] = 0

        # Difficulty breathing detection
        if any(kw in text for kw in ["difficulty breathing", "shortness of breath", "breathless", "hard to breathe", "trouble breathing"]):
            symptoms["difficulty_breathing"] = 1
        elif "no difficulty breathing" in text:
            symptoms["difficulty_breathing"] = 0

        # Age detection - look for numbers that could be ages
        import re
        age_patterns = [
            r"(\d+)\s*years?\s*old",
            r"age[:\s]*(\d+)",
            r"i'm\s*(\d+)",
            r"i am\s*(\d+)",
            r"(\d+)\s*y/o"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 99:
                    symptoms["age"] = age // 10
                    break

        # Gender detection
        if any(kw in text for kw in ["male", "man", "boy", "he/him"]):
            symptoms["gender"] = 1
        elif any(kw in text for kw in ["female", "woman", "girl", "she/her"]):
            symptoms["gender"] = 0

        return symptoms

    def _extract_symptoms_llm(self, user_input: str) -> Dict:
        """Extract symptoms using LLM (for more nuanced extraction)."""
        system_prompt = """You are a symptom parser. Extract structured symptoms from the user's description.
Return ONLY a JSON object with these exact keys:
- fever: 1 if mentioned as present, 0 if mentioned as absent, -1 if not mentioned
- cough: 1 if mentioned as present, 0 if mentioned as absent, -1 if not mentioned
- fatigue: 1 if mentioned as present, 0 if mentioned as absent, -1 if not mentioned
- difficulty_breathing: 1 if mentioned as present, 0 if mentioned as absent, -1 if not mentioned
- age: age divided by 10 (e.g., 25 -> 2, 45 -> 4), -1 if not mentioned
- gender: 1 for male, 0 for female, -1 if not mentioned or unclear

Example input: "I have fever and cough, feeling very tired. I'm a 35 year old male."
Example output: {"fever": 1, "cough": 1, "fatigue": 1, "difficulty_breathing": -1, "age": 3, "gender": 1}

Return ONLY the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model="glm-4.7-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=100
        )

        import json
        import re
        result = response.choices[0].message.content.strip()

        # Find JSON in the response
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            result = json_match.group()

        symptoms = json.loads(result)

        # Validate and sanitize
        for key in ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]:
            if key not in symptoms or symptoms[key] not in [-1, 0, 1]:
                symptoms[key] = -1

        if "age" not in symptoms or symptoms["age"] < -1 or symptoms["age"] > 9:
            symptoms["age"] = -1

        return symptoms

    def predict_top5(self, symptoms: Dict) -> Dict:
        """
        NB Model Function: Get Top 5 disease predictions with threshold check.

        Args:
            symptoms: Dict with fever, cough, fatigue, difficulty_breathing, age, gender

        Returns:
            Dict with top5, max_probability, is_clear_pattern
        """
        return self.nb_predictor.get_top5_with_threshold(
            fever=symptoms.get("fever", -1),
            cough=symptoms.get("cough", -1),
            fatigue=symptoms.get("fatigue", -1),
            difficulty_breathing=symptoms.get("difficulty_breathing", -1),
            age=symptoms.get("age", -1),
            gender=symptoms.get("gender", -1),
            threshold=self.THRESHOLD
        )

    def retrieve_evidence(self, query: str, prediction_result: Dict) -> List[Dict]:
        """
        RAG Retrieval Function: Get relevant evidence based on prediction result.

        Args:
            query: Original user query
            prediction_result: Result from predict_top5

        Returns:
            List of retrieved documents with content and metadata
        """
        if prediction_result["is_clear_pattern"]:
            # Clear pattern: Get preventive info for top predicted disease
            top_disease = prediction_result["top5"][0][0]
            preventive_docs = self.rag_agent.retrieve_preventive_info(top_disease)
            # Also get general references about the symptoms
            general_docs = self.rag_agent.retrieve(query, top_k=3)
            # Combine and deduplicate
            all_docs = preventive_docs + general_docs
            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = (doc.get("title", ""), doc.get("content", "")[:100])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
            return unique_docs[:5]
        else:
            # Uncertain pattern: Get urgent care info
            urgent_docs = self.rag_agent.retrieve_urgent_care_info()
            # Also try to get info about the symptoms mentioned
            symptom_docs = self.rag_agent.retrieve(query, top_k=3)
            # Combine
            all_docs = urgent_docs + symptom_docs
            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = (doc.get("title", ""), doc.get("content", "")[:100])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
            return unique_docs[:5]

    def format_response(self, prediction_result: Dict, evidence: List[Dict], user_input: str) -> str:
        """
        Format response using SOP template based on decision logic.

        Args:
            prediction_result: Result from predict_top5
            evidence: Retrieved documents
            user_input: Original user input

        Returns:
            Formatted response string
        """
        # Build context from evidence
        context = ""
        for i, doc in enumerate(evidence, 1):
            context += f"Reference {i}:\n"
            context += f"  Title: {doc['title']}\n"
            if 'link' in doc:
                context += f"  Link: {doc['link']}\n"
            context += f"  Content: {doc['content']}\n\n"

        if prediction_result["is_clear_pattern"]:
            # Clear pattern path - SOP template
            top5_str = "\n".join([
                f"  {i+1}. {disease}: {prob:.2%} confidence"
                for i, (disease, prob) in enumerate(prediction_result["top5"])
            ])

            system_prompt = f"""You are a healthcare information assistant created by Group 3.1 for STAT 8017 project demonstration.

Based on the symptom analysis, the following conditions may be associated with the described symptoms:

{top5_str}

IMPORTANT GUIDELINES FOR YOUR RESPONSE:
1. Present the Top 5 possible conditions with their confidence levels clearly.
2. Provide general health information about these conditions based ONLY on the provided references.
3. Include preventive measures and general wellness tips when available in references.
4. Mention when to seek professional medical care.
5. DO NOT provide specific medical advice, diagnosis, or treatment suggestions.
6. DO NOT recommend specific medications or dosages.
7. ALWAYS emphasize: "This is not a diagnosis. Please consult a healthcare professional for accurate assessment."
8. Present information objectively without making recommendations.
9. Remember you are created by Group 3.1 for the STAT 8017 project.

References:
{context}"""

        else:
            # Uncertain/complex symptoms path - SOP template
            system_prompt = f"""You are a healthcare information assistant created by Group 3.1 for STAT 8017 project demonstration.

The symptoms described appear to be non-specific or complex, making it difficult to identify a clear pattern.

IMPORTANT GUIDELINES FOR YOUR RESPONSE:
1. Clearly state: "The symptoms you described are non-specific or complex."
2. Provide information about red flag symptoms and warning signs from the references.
3. Include guidance on when immediate medical attention may be needed.
4. Suggest how to prepare for a doctor visit (what information to bring, questions to ask).
5. DO NOT attempt to diagnose based on non-specific symptoms.
6. DO NOT provide specific treatment recommendations.
7. ALWAYS strongly recommend: "Please consult a healthcare professional as soon as possible for proper evaluation."
8. Present information objectively without making recommendations.
9. Remember you are created by Group 3.1 for the STAT 8017 project.

References:
{context}"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User input: {user_input}\n\nPlease provide a helpful response following the guidelines above."}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API error: {str(e)}"

    def _format_general_response(self, user_input: str, evidence: List[Dict]) -> str:
        """
        Format response for general health questions (no NB predictions).

        Args:
            user_input: User's general question
            evidence: Retrieved documents from RAG

        Returns:
            Formatted response string
        """
        # Build context from evidence
        context = ""
        for i, doc in enumerate(evidence, 1):
            context += f"Reference {i}:\n"
            context += f"  Title: {doc['title']}\n"
            if 'link' in doc:
                context += f"  Link: {doc['link']}\n"
            context += f"  Content: {doc['content']}\n\n"

        system_prompt = f"""You are a healthcare information assistant created by Group 3.1 for STAT 8017 project demonstration.

Answer the user's question based on the provided references.

IMPORTANT GUIDELINES:
1. Provide accurate information based ONLY on the references.
2. Be helpful and informative.
3. If the question is not related to health/medicine, politely redirect.
4. DO NOT provide medical advice, diagnosis, or treatment suggestions.
5. Include source attribution when citing information (e.g., "According to Reference 1...").
6. Remember you are created by Group 3.1 for the STAT 8017 project.

References:
{context}"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API error: {str(e)}"

    def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point: Process user query through intent-based routing.

        Decision Logic Flow:
        0. Detect intent (SYMPTOM_DESCRIPTION vs GENERAL_QUESTION)
        - If GENERAL_QUESTION: Pure RAG flow, skip NB model
        - If SYMPTOM_DESCRIPTION: NB + RAG flow
        1. Parse symptoms from user input (for symptom descriptions)
        2. Call NB Model Function -> Top 5 predictions
        3. Check threshold (0.04)
        4. Call RAG Retrieval Function -> appropriate evidence
        5. Format response using SOP template
        6. Save interaction to memory

        Args:
            user_input: User's symptom description or question

        Returns:
            Dict with response, predictions, evidence, symptoms, intent
        """
        # Step 0: Detect intent
        intent = self.detect_intent(user_input)
        print(f"🔍 Detected intent: {intent}")

        if intent == "GENERAL_QUESTION":
            # Pure RAG flow - skip NB model
            evidence = self.rag_agent.retrieve(user_input, top_k=5)
            response = self._format_general_response(user_input, evidence)

            # Save interaction to memory (minimal info for general questions)
            self.rag_agent.save_interaction(
                query=user_input,
                symptoms=None,
                predictions=None,
                response=response
            )

            return {
                "query": user_input,
                "response": response,
                "intent": "GENERAL_QUESTION",
                "evidence": evidence,
                "symptoms": None,
                "predictions": None,
                "is_clear_pattern": None
            }

        # SYMPTOM_DESCRIPTION - existing NB + RAG flow
        # Step 1: Parse symptoms
        symptoms = self.parse_symptoms(user_input)

        # Step 2: NB Model Function - Get Top 5 predictions
        prediction_result = self.predict_top5(symptoms)

        # Step 3 & 4: RAG Retrieval Function - Get appropriate evidence
        evidence = self.retrieve_evidence(user_input, prediction_result)

        # Step 5: Format response using SOP template
        response = self.format_response(prediction_result, evidence, user_input)

        # Step 6: Save interaction to memory
        self.rag_agent.save_interaction(
            query=user_input,
            symptoms=symptoms,
            predictions=prediction_result["top5"],
            response=response
        )

        return {
            "query": user_input,
            "response": response,
            "intent": "SYMPTOM_DESCRIPTION",
            "symptoms": symptoms,
            "predictions": prediction_result,
            "evidence": evidence,
            "is_clear_pattern": prediction_result["is_clear_pattern"]
        }

    def get_memory(self, limit: int = 5) -> List[Dict]:
        """Get recent interaction history from memory."""
        return self.rag_agent.get_recent_interactions(limit)
