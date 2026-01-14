import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Now import your other libraries
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

class CrediTrustEngine:
    def __init__(self, index_path):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatMistralAI(model="open-mistral-nemo", temperature=0.1)
        self.vector_store = self._load_vector_store()
        self.rag_chain = self._create_chain()

    def _load_vector_store(self):
        """Robust loading with error handling."""
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index directory {self.index_path} missing.")
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"❌ Critical Error: Could not load Vector Store: {e}")
            return None

    def _create_chain(self):
        """Builds the RAG pipeline."""
        if not self.vector_store:
            return None
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        prompt = ChatPromptTemplate.from_template("""
        You are a Senior Strategic Analyst. Answer based ONLY on the context. 
        Cite the 'complaint_id' for every fact.
        Context: {context}
        Question: {input}
        """)
        
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)

    def run_query(self, user_input):
        """The main entry point with error safety."""
        if not self.rag_chain:
            return "Engine Error: Knowledge base unavailable.", []
        
        try:
            response = self.rag_chain.invoke({"input": user_input})
            # Return both answer and source documents for transparency
            return response["answer"], response["context"]
        except Exception as e:
            return f"⚠️ API Error: Unable to process request. (Details: {str(e)})", []

if __name__ == "__main__":
    # This only runs if you type 'python src/rag_engine.py'
    # It's great for a quick "smoke test"
    test_engine = CrediTrustEngine("vector_store/faiss_index_sample")
    print(test_engine.run_query("Test query"))