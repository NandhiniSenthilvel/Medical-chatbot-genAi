🩺 Medical Chatbot GenAI

An end-to-end Generative AI Medical Chatbot developed using advanced Retrieval-Augmented Generation (RAG). The system processes medical literature, generates semantic vector embeddings, and stores them securely within a cloud-based vector database to provide accurate, context-aware clinical responses.


🚀 Features
* **Semantic Document Ingestion:** Programmatically extracts and parses text structures from comprehensive medical literature PDFs.
* **Efficient Vector Storage:** Utilizes chunks optimized for serverless storage processing.
* **Cloud Ingestion Pipeline:** Integrates a secure streaming process to upsert vector matrices directly via the native Pinecone SDK.
* **Context-Driven Responses:** Leverages advanced RAG architecture to anchor responses strictly to verified medical data, eliminating model hallucinations.

---

## 🏗️ System Architecture
* **Core Framework:** LangChain / LangChain-Community
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions)
* **Vector Database:** Pinecone (Cloud Serverless Index)
* **LLM Engine:** OpenAI `gpt-4o-mini`

---

## 💻 Setup Instructions

Follow these steps to set up the development environment on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/NandhiniSenthilvel/Medical-chatbot-genAi.git](https://github.com/NandhiniSenthilvel/Medical-chatbot-genAi.git)
cd Medical-chatbot-genAi

**2. Create and Activate the Environment**

# Verify your conda installation first
conda --version

# Create a isolated virtual environment using Python 3.10
conda create -n medibot python=3.10 -y

# Activate the active workspace environment
conda activate medibot

**3. Install Dependencies**
pip install -r requirements.txt
