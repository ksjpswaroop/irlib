import os
import json
from irlib import BM25Retriever, DenseRetriever, SearchPipeline, StreamingDocumentProcessor

# Note: In a real scenario, we'd download the dataset files here.
# For this script, we'll simulate the ingestion from a hypothetical local path.
dataset_path = "/Users/swaroop/irlib/data/indian-judiciary/chunks.jsonl"

def benchmark_on_judiciary_data():
    print("Initializing components...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    pipeline = SearchPipeline(lexical_retriever=bm25, dense_retriever=dense)
    processor = StreamingDocumentProcessor(batch_size=1000)

    # Simulated data stream
    def get_docs():
        # This simulates reading the large file line by line
        for i in range(5000): # Test with 5000 chunks for now
            yield f"Legal judgment excerpt {i}: The court held that the bail condition..."

    print("Indexing using StreamingDocumentProcessor...")
    # Using the streaming processor to index
    # We pass the generator to the processor
    processor.process(get_docs(), retriever=pipeline.lexical)
    # Dense retrieval indexing would similarly be batched

    print("Indexing complete. Running sample query...")
    query = "bail condition POCSO"
    results = pipeline.search(query, top_k=5)
    
    print(f"Results for '{query}':")
    for doc_id, score in results:
        print(f"DocID: {doc_id}, Score: {score}")

if __name__ == "__main__":
    benchmark_on_judiciary_data()
