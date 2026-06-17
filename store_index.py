import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Load active environment variables from .env file configuration
load_dotenv()

# Explicitly set fallback cloud credentials configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_5EM81u_88V4e3B9chnLx3kY1qEfnm89UuJRRPDVSFtaM74h59vnaNYSMJxL2PxD5B8v3ox")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

DATA_DIR = "data/"
INDEX_NAME = "testbot"

print("--- Starting Clean Native Pipeline ---")

# 2. Extract text data from your local PDF data directory path strings
print("Step 1: Extracting text from PDF data source...")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"[⚠️ WARNING] Data folder '{DATA_DIR}' was empty or missing. Please paste your medical PDF inside it.")

loader = PyPDFDirectoryLoader(DATA_DIR)
raw_documents = loader.load()
print(f"-> Successfully extracted {len(raw_documents)} total pages.")

# 3. Fragment text documents frames into optimized semantic chunk tokens
print("Step 2: Splitting text document frames into semantic chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=20
)
text_chunks = text_splitter.split_documents(raw_documents)
print(f"-> Total text chunks generated: {len(text_chunks)}")

# 4. Initialize the Hugging Face dense vector transformer embedding engine
print("Step 3: Initializing embedding transformer client engine...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Connect to the active Pinecone vector database service infrastructure
print("Step 4: Connecting to Pinecone Cloud Database...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Dynamically construct the cloud cluster index if it is missing or not deployed
existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"-> Index '{INDEX_NAME}' not found. Creating it automatically...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Matches dimensions outputted by all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("-> Index created and ready!")
else:
    print(f"-> Index '{INDEX_NAME}' already exists. Preparing vector synchronization...")

# 6. Stream chunk values into the active cloud pipeline vector store instance
print("Step 5: Vectorizing and streaming chunks directly via Native Pinecone SDK...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("🚀 --- Pipeline Complete! Pinecone Vector Store initialized successfully! ---")