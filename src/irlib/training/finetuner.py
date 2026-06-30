"""
LegalFineTuner: Domain adaptation for information retrieval.
Optimizes embedding models using contrastive loss on legal corpora.
"""
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

class LegalFineTuner:
    """Fine-tunes transformer embeddings on legal data (e.g., bhasha_legal)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def prepare_examples(self, pairs: List[Tuple[str, str, float]]):
        """Pairs of (text1, text2, relevance_score)."""
        return [InputExample(texts=[p[0], p[1]], label=p[2]) for p in pairs]

    def fine_tune(self, train_examples: List[InputExample], output_path: str, epochs: int = 1):
        """Train using MultipleNegativesRankingLoss."""
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            output_path=output_path,
            warmup_steps=100
        )
        print(f"Fine-tuning complete. Model saved to: {output_path}")
