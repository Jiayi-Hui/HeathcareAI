# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import List, Dict
from bert_score import BERTScorer
import nltk
from metric.evaluator import get_evaluator
from utils import convert_to_json

nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HealthcareEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the official UniEval dialogue evaluator
        try:
            self.unieval_dialogue = get_evaluator('dialogue', device=self.device)
        except Exception as e:
            print(f"Warning: Could not initialize UniEval: {e}")
            self.unieval_dialogue = None

        # Initialize BERTScorer
        try:
            self.bert_scorer = BERTScorer(lang="en", device=self.device)
        except Exception as e:
            print(f"Warning: Could not initialize BERTScorer: {e}")
            self.bert_scorer = None
            
    def get_bert_score(self, prediction: str, ground_truth: str) -> float:
        """Calculates semantic similarity using BERTScore."""
        if not ground_truth:
            return 0.0
        try:
            P, R, F1 = self.bert_scorer.score([prediction], [ground_truth])
            return F1.item()
        except Exception:
            return 0.0

    def calculate_grounding_score(self, prediction: str, retrieved_docs: List[Dict]) -> float:
        """
        Measures how much of the response is grounded in the retrieved documents.
        This acts as a lightweight proxy for factuality.
        """
        context = " ".join([doc.get('content', '') for doc in retrieved_docs]).lower()
        sentences = [s.strip() for s in prediction.split('.') if len(s.strip()) > 10]
        
        if not sentences:
            return 0.0
            
        supported = 0
        for sent in sentences:
            # Check if significant words in the sentence appear in the context
            words = [w for w in sent.lower().split() if len(w) > 4]
            if words and any(word in context for word in words):
                supported += 1
                
        return supported / len(sentences)

   def get_unieval_metrics(self, prediction: str, query: str, retrieved_docs: List[Dict]) -> Dict[str, float]:
        """
        Official UniEval implementation for the 'dialogue' task.
        Evaluates Naturalness, Coherence, Engagedness, and Groundedness.
        """
        if self.unieval_dialogue is None:
            return {"error": "UniEval model not loaded"}

        # 1. Prepare inputs
        # src_list: The dialogue history/user prompt
        # context_list: The retrieved knowledge the model should use
        # output_list: The model's generated response
        src_list = [query]
        context_text = " ".join([doc.get('content', '') for doc in retrieved_docs])
        context_list = [context_text]
        output_list = [prediction]

        # 2. Convert to UniEval JSON format
        data = convert_to_json(
            output_list=output_list, 
            src_list=src_list, 
            context_list=context_list
        )

        # 3. Evaluate
        try:
            eval_scores = self.unieval_dialogue.evaluate(data, print_result=False)
            # The evaluator returns a list of dictionaries; we take the first result
            return eval_scores[0]
        except Exception as e:
            print(f"UniEval Scoring Error: {e}")
            return {}
