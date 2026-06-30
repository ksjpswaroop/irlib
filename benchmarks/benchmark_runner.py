import json
import time
from irlib import SearchPipeline, BM25Retriever, DenseRetriever
from irlib.streaming import StreamingDocumentProcessor

def load_registry(path="/Users/swaroop/irlib/benchmarks/datasets/registry.json"):
    with open(path, 'r') as f:
        return json.load(f)

def run_large_scale_benchmark(dataset_name):
    print(f"--- Starting Benchmark: {dataset_name} ---")
    registry = load_registry()
    dataset_info = registry["benchmarks"][dataset_name]
    
    # Initialize pipeline
    pipeline = SearchPipeline(lexical_retriever=BM25Retriever(), dense_retriever=DenseRetriever())
    processor = StreamingDocumentProcessor(batch_size=5000)
    
    print(f"Orchestrating retrieval over {dataset_info['size_chunks']} chunks...")
    
    # Simulated Streaming Load for massive scale
    def document_generator():
        # In production, replace with actual streaming file reader (e.g. gzip)
        for i in range(dataset_info['size_chunks']):
            yield f"Context block {i} from domain {dataset_name}"
            
    # Batch indexing via streaming processor
    start_time = time.perf_counter()
    processor.process(document_generator(), retriever=pipeline.lexical)
    index_time = time.perf_counter() - start_time
    
    print(f"Indexing completed in {index_time:.2f}s")
    
    # Run a representative query
    query = "legal bail conditions"
    search_start = time.perf_counter()
    results = pipeline.search(query, top_k=10)
    search_time = time.perf_counter() - search_start
    
    return {
        "dataset": dataset_name,
        "index_time": index_time,
        "search_time": search_time,
        "throughput": dataset_info['size_chunks'] / index_time
    }

if __name__ == "__main__":
    for ds in ["indian_judiciary"]: # Test on judiciary first
        metrics = run_large_scale_benchmark(ds)
        print(f"Metrics: {metrics}")
