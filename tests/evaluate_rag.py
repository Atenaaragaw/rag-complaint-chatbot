import sys
import os
# Add src to path so we can import our engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag_engine import CrediTrustEngine

# 1. Define 'Golden' Test Questions & Expected Insights
test_cases = [
    {
        "query": "Are there issues with late payment reporting?",
        "expected_keyword": "late payment",
        "category": "Credit Reporting"
    },
    {
        "query": "What are customers saying about billing transparency?",
        "expected_keyword": "bill",
        "category": "Billing"
    }
]

def run_evaluation():
    engine = CrediTrustEngine("vector_store/faiss_index_sample")
    print("🧪 Starting RAG Quality Audit...\n")
    
    for i, test in enumerate(test_cases):
        print(f"Test {i+1} [{test['category']}]: {test['query']}")
        answer, sources = engine.run_query(test['query'])
        
        # Metric 1: Groundedness (Did we get sources?)
        source_count = len(sources)
        status = "✅ PASS" if source_count > 0 and test['expected_keyword'] in answer.lower() else "❌ FAIL"
        
        print(f"Result: {status}")
        print(f"Sources Found: {source_count}")
        print("-" * 20)

if __name__ == "__main__":
    run_evaluation()