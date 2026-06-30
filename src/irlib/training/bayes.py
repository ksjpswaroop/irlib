"""
BayesianDocumentRouter for domain-aware retrieval.
Uses Naive Bayes to classify query intent into specific index shards.
"""
import math
import re
import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any

class BayesianDocumentRouter:
    """
    Routes queries to the most relevant index shard using Multinomial Naive Bayes.
    Traceability: Logs confidence distribution per domain.
    """
    def __init__(self):
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.domains: Set[str] = set()
        self.vocabulary: Set[str] = set()

    def train(self, corpus_by_domain: Dict[str, List[str]]):
        total_docs = sum(len(docs) for docs in corpus_by_domain.values())
        for domain, docs in corpus_by_domain.items():
            self.domains.add(domain)
            self.priors[domain] = len(docs) / total_docs
            
            # Count word occurrences with Laplace smoothing (1.0)
            word_counts = Counter()
            for doc in docs:
                tokens = re.findall(r"\b\w+\b", doc.lower())
                word_counts.update(tokens)
                self.vocabulary.update(tokens)
            
            total_words = sum(word_counts.values())
            for word in self.vocabulary:
                self.likelihoods[domain][word] = (word_counts[word] + 1) / (total_words + len(self.vocabulary))

    def route(self, query: str) -> Dict[str, Any]:
        """Routes query with trace and confidence."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        scores = {}
        
        for domain in self.domains:
            log_prob = math.log(self.priors[domain])
            for token in tokens:
                # Use likelihood if known, else minimal epsilon
                prob = self.likelihoods[domain].get(token, 1.0 / (len(self.vocabulary) + 1))
                log_prob += math.log(prob)
            scores[domain] = log_prob
        
        # Softmax-like confidence
        total = sum(math.exp(s) for s in scores.values())
        confidence = {d: math.exp(s) / total for d, s in scores.items()}
        chosen = max(confidence, key=confidence.get)
        
        return {
            "chosen_domain": chosen,
            "confidence": confidence,
            "traceability_id": hash(query + str(datetime.datetime.now()))
        }
