import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def run_indexing():
    print("--- Loading Processed Data ---")
    df = pd.read_csv('data/processed/filtered_complaints.csv')

    # 1. Stratified Sampling (15,000 complaints)
    print("--- Performing Stratified Sampling ---")
    sample_size = 15000
    # Stratify by 'Product' to keep representation balanced
    df_sample, _ = train_test_split(
        df, 
        train_size=sample_size, 
        stratify=df['Product'], 
        random_state=42
    )

    # 2. Text Chunking
    print("--- Chunking Narratives ---")
    # Specs: 500 chars size, 50 chars overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = []
    metadata = []

    for _, row in df_sample.iterrows():
        narrative = row['cleaned_narrative']
        doc_chunks = text_splitter.split_text(narrative)
        
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            # Metadata allows Asha to trace answers back to specific IDs/Products
            metadata.append({
                "complaint_id": row['Complaint ID'],
                "product": row['Product'],
                "issue": row['Issue'],
                "chunk_index": i
            })

    # 3. Embedding and Vector Store
    print(f"--- Generating Embeddings for {len(chunks)} chunks ---")
    # This model is lightweight and fast for local CPUs
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        metadatas=metadata
    )

    # 4. Save the Index
    print("--- Saving Vector Store ---")
    vector_store.save_local("vector_store/faiss_index_sample")
    print("Success! Vector store saved in 'vector_store/faiss_index_sample'")

if __name__ == "__main__":
    run_indexing()