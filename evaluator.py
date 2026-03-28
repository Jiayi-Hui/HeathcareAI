# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import List, Dict
from bert_score import score as bert_score_func

class HealthcareEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_bert_score(self, prediction: str, ground_truth: str) -> float:
        """Calculates semantic similarity using BERTScore."""
        if not ground_truth:
            return 0.0
        try:
            P, R, F1 = bert_score_func([prediction], [ground_truth], lang="en", device=self.device)
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

    def get_unieval_metrics(self, prediction: str) -> Dict[str, float]:
        """
        UniEval-inspired metrics for text quality. 
        Calculates Coherence and Fluency based on structural markers.
        """
        # Simple heuristic-based UniEval proxy
        sentences = prediction.split('.')
        word_count = len(prediction.split())
        
        coherence = 1.0 if len(sentences) > 1 else 0.5
        fluency = min(1.0, word_count / 20) # Penalize extremely short, clipped answers
        
        return {
            "coherence": coherence,
            "fluency": fluency,
            "consistency": 1.0 # Placeholder for logical check
        }
