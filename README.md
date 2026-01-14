🤖 Intelligent Complaint Analysis for Financial Services
Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

1. 📈 Project Overview
This project transforms 357,000+ unstructured customer complaints into a strategic asset for CrediTrust Financial. By leveraging Retrieval-Augmented Generation (RAG), we enable stakeholders to identify systemic issues and product trends in seconds using natural language queries.

2. 🎯 Core Objectives
Speed: Reduced manual trend identification time from days to seconds using semantic search.

Accessibility: Empowered non-technical teams (Support/Compliance) to query complex datasets via a Streamlit interface.

Proactivity: Shifted from reactive firefighting to proactive issue resolution through automated strategic reporting.

3. 🛠️ Technical Implementation (Final)
Task 1 & 2: Data Engineering & Vector Infrastructure
Corpus Processing: Isolated 357,284 relevant complaints across Credit Cards, Loans, and Savings.

Semantic Indexing: Generated 44,052 vector embeddings using the all-MiniLM-L6-v2 model.

Vector Store: Powered by FAISS for millisecond-latency retrieval with a stratified 15,000-complaint sample to ensure proportional representation.

Granularity: Implemented 500-character chunks with a 50-character overlap to preserve semantic context for the LLM.

Task 3: Strategic RAG Engine
LLM Engine: Integrated Mistral-Nemo-Instruct-2407 via native API for high-reasoning stability.

Strategic Prompting: Developed a "Senior Analyst" persona that enforces grounded responses and mandatory citations of Complaint IDs.

Stability Fix: Pivoted from the Hugging Face Router to a native Mistral integration to resolve 2026 infrastructure "Provider/Task" validation hurdles.

Task 4: Interactive Analyst Dashboard
Interface: Built a custom Streamlit dashboard featuring real-time analysis generation.

Transparency: Included a "Reference Data" section using expandable sources to allow analysts to verify AI insights against raw complaint text.

4. 📂 Final Project Structure
Plaintext

rag-complaint-chatbot/
├── data/
│   ├── raw/                # Original CFPB Data
│   └── processed/          # Cleaned CSVs
├── notebooks/
│   ├── 01_eda.ipynb        # Data cleaning & visualization
│   └── 02_test_rag.ipynb   # RAG prototyping & validation
├── src/
│   ├── preprocessing.py    # CFPB noise removal logic
│   ├── indexing.py         # FAISS vector creation
│   └── rag_engine.py       # Final Mistral-Nemo RAG logic
├── vector_store/           # Persisted FAISS Index (index.faiss/index.pkl)
├── tests/                  # Automated integrity checks
├── app.py                  # Streamlit Dashboard Interface
└── requirements.txt        # Verified 2026 dependencies
5. ⚙️ CI/CD & Reliability
Environment Sync: Validates dependencies from requirements.txt.

Integrity Tests: tests/test_rag_core.py verifies metadata consistency and FAISS index health before deployment.

Provider Pinning: Forced hf-inference and native API routes to bypass third-party server instability.

6. 📊 Key Strategic Insights Generated
Identified Trend: Recurring "Late Payment Remarks" on accounts despite verified timely payments.

Root Cause: Discovered a potential synchronization gap between payment processing systems and credit bureau reporting.

Recommendation: Immediate audit of the automated reporting triggers for the Savings and Personal Loan divisions.

🚀 To Run the Project:
Install requirements: pip install -r requirements.txt

Configure .env with your MISTRAL_API_KEY.

Launch Dashboard: streamlit run app.py