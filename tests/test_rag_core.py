import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def test_vector_store_integrity():
    """Check if the FAISS index exists and can be searched."""
    index_path = "vector_store/faiss_index_sample"
    assert os.path.exists(index_path), "FAISS index directory missing!"
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Perform a dummy search
    results = vector_store.similarity_search("credit card billing issue", k=1)
    
    assert len(results) > 0, "Vector store returned no results!"
    assert "product" in results[0].metadata, "Metadata missing from chunks!"
    print("✅ Vector Store Integrity Check Passed!")

if __name__ == "__main__":
    test_vector_store_integrity()