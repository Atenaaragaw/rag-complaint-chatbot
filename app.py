import streamlit as st
from src.rag_engine import get_rag_chain

# 1. Page Configuration
st.set_page_config(
    page_title="CrediTrust Strategic Analyst",
    page_icon="🏦",
    layout="wide"
)

# 2. Cached Engine Loader
# This ensures we don't reload the FAISS index on every click
@st.cache_resource
def load_rag_engine():
    return get_rag_chain()

try:
    # Initialize the engine
    rag_chain = load_rag_engine()

    # 3. Sidebar for System Status
    with st.sidebar:
        st.title("⚙️ System Status")
        st.success("FAISS Index Loaded")
        st.success("Mistral-Nemo Online")
        st.divider()
        st.info("Role: Senior Strategic Analyst")

    # 4. Main Dashboard UI
    st.title("🏦 CrediTrust: Customer Complaint Analysis")
    st.markdown("Use this AI-powered dashboard to investigate strategic trends in customer complaints.")

    # Search Interface
    query = st.text_input(
        "Enter investigation query:",
        placeholder="e.g., Identify recurring billing errors or late payment disputes..."
    )

    if st.button("Generate Strategic Report"):
        if query:
            with st.spinner("Analyzing complaint database..."):
                # Execute the RAG chain
                response = rag_chain.invoke({"input": query})
                
                # Layout for Results
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("💡 Strategic Insights")
                    st.markdown(response["answer"])

                with col2:
                    st.subheader("🔍 Referenced Sources")
                    # 'context' contains the documents retrieved by FAISS
                    for i, doc in enumerate(response["context"]):
                        # Extract metadata if available, otherwise use index
                        cid = doc.metadata.get('complaint_id', f"Source {i+1}")
                        with st.expander(f"Complaint ID: {cid}"):
                            st.write(doc.page_content)
        else:
            st.warning("Please enter a query to begin analysis.")

except Exception as e:
    st.error(f"⚠️ Critical System Error: {e}")
    st.info("Check your .env file for the MISTRAL_API_KEY.")