import json
from src.rag_engine import CrediTrustEngine

def evaluate():
    engine = CrediTrustEngine("vector_store/faiss_index_sample")
    with open("tests/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    print(f"📋 Running Evaluation on {len(ground_truth)} cases...\n")

    for case in ground_truth:
        answer, sources = engine.run_query(case["question"])
        
        # Check if the correct Complaint ID was retrieved
        retrieved_ids = [doc.metadata.get('complaint_id') for doc in sources]
        id_match = case["context_id"] in retrieved_ids
        
        # Check if the expected answer keyword is in the LLM response
        content_match = case["expected_answer"] in answer

        print(f"Q: {case['question']}")
        print(f"ID Match: {'✅' if id_match else '❌'} | Content Match: {'✅' if content_match else '❌'}")
        print("-" * 40)

if __name__ == "__main__":
    evaluate()