import os
from dotenv import load_dotenv

# Using the native Mistral integration for maximum stability
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_rag_chain():
    # 1. Embeddings (Keep using local HF embeddings to save credits)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load FAISS Vector Store
    vector_store = FAISS.load_local(
        "vector_store/faiss_index_sample", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 3. Use Native Mistral AI (Bypasses Hugging Face Router Errors)
    llm = ChatMistralAI(
        model="open-mistral-nemo", 
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.1
    )

    # 4. Senior Strategic Analyst Prompt
    system_prompt = (
        "You are a Senior Strategic Analyst for CrediTrust. "
        "Use ONLY the following context to answer the user's question. "
        "Always cite the 'Complaint ID' for every fact provided.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build the Final RAG Chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

if __name__ == "__main__":
    print("🚀 Initializing Native Mistral RAG Engine...")
    try:
        chain = get_rag_chain()
        test_q = "What specific billing errors are being reported?"
        result = chain.invoke({"input": test_q})
        
        print("\n" + "="*50)
        print("📊 ANALYSIS REPORT")
        print("="*50)
        print(result["answer"])
        print("="*50)
    except Exception as e:
        print(f"\n❌ Final Error: {e}")