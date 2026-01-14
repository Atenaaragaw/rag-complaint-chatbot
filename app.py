import streamlit as st
import os
from src.rag_engine import CrediTrustEngine

# 1. Page Configuration
st.set_page_config(page_title="CrediTrust Strategic Analyst", layout="wide")

st.title("🏦 CrediTrust Strategic Complaint Intelligence")
st.markdown("---")

# 2. Initialize the Engine (Using caching to prevent reloading on every click)
@st.cache_resource
def init_engine():
    # Points to the root-level vector store
    index_path = os.path.join(os.getcwd(), "vector_store", "faiss_index_sample")
    return CrediTrustEngine(index_path)

try:
    engine = init_engine()
    st.sidebar.success("✅ RAG Engine Online")
except Exception as e:
    st.sidebar.error(f"❌ Engine Offline: {e}")
    st.stop()

# 3. User Interface
query = st.text_input("Enter a strategic question (e.g., 'What are the main issues with billing?')")

if query:
    with st.spinner("Analyzing high-volume complaint data..."):
        # We now use the .run_query() method from our new class
        answer, sources = engine.run_query(query)
        
        # Display the Analysis
        st.subheader("📊 Strategic Analysis Report")
        st.info(answer)
        
        # 4. Source Transparency (Addressing feedback on 'robustness')
        st.subheader("📂 Reference Data")
        if sources:
            for doc in sources:
                with st.expander(f"Complaint ID: {doc.metadata.get('complaint_id', 'Unknown')}"):
                    st.write(f"**Product:** {doc.metadata.get('product')}")
                    st.write(f"**Issue:** {doc.metadata.get('issue')}")
                    st.write(f"**Content:** {doc.page_content}")
        else:
            st.warning("No specific source documents were retrieved for this query.")

# Sidebar Information
st.sidebar.markdown("---")
st.sidebar.info("""
**System Specs:**
- **Model:** Mistral-Nemo-Instruct
- **Retriever:** FAISS (all-MiniLM-L6-v2)
- **Status:** Operational
""")