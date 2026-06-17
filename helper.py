import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class MedicalBotPipeline:
    def __init__(self):
        """
        Initializes core configurations, maps global state variables, 
        and defines system prompt guidelines.
        """
        self.index_name = "testbot"
        
        # 1. Fallback environment configuration validations
        os.environ["PINECONE_API_KEY"] = os.getenv(
            "PINECONE_API_KEY", 
            "pcsk_5EM81u_88V4e3B9chnLx3kY1qEfnm89UuJRRPDVSFtaM74h59vnaNYSMJxL2PxD5B8v3ox"
        )
        os.environ["GEMINI_API_KEY"] = os.getenv(
            "GEMINI_API_KEY", 
            "AQ.Ab8RN6Jw4Qu7ud3leyhX0TzI2MQxw-nV9yLOUSB8DKmmDklE8g"
        )

        # 2. Setup the prompt layout constraints 
        self.system_prompt = (
            "You are a helpful and professional medical assistant. Use the following pieces of "
            "retrieved context to answer the user's question accurately.\n"
            "If you do not know the answer, or if the context doesn't contain it, say clearly that you "
            "do not know. Do not try to make up an answer.\n\n"
            "Retrieved Context:\n{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        # Runtime components placeholder
        self.retriever = None
        self.llm = None
        
    def initialize_pipeline(self):
        """
        Connects to Pinecone and instantiates the Gemini API engine instance.
        """
        try:
            # Load embeddings model layout maps
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            
            # Mount Pinecone Vector Storage connections
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name, 
                embedding=embeddings
            )
            self.retriever = docsearch.as_retriever(search_kwargs={"k": 3})
            
            # Setup stable production-tier inference model endpoint
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0.4, 
                max_output_tokens=500
            )
            return True
        except Exception as e:
            print(f"[ERROR] Pipeline initialization failed: {str(e)}")
            return False

    def generate_response(self, user_query: str) -> str:
        """
        Executes native text contextual retrieval and streams results through LLM instances.
        """
        if not self.retriever or not self.llm:
            raise RuntimeError("Pipeline must be initialized via .initialize_pipeline() before processing.")
            
        try:
            # Look up semantic match blocks inside Pinecone
            matching_docs = self.retriever.invoke(user_query)
            combined_context = "\n\n".join([doc.page_content for doc in matching_docs])
            
            # Map parameters straight to formatted system message structures
            formatted_messages = self.prompt.format_messages(
                context=combined_context, 
                input=user_query
            )
            
            # Request response text generation block from Gemini
            response = self.llm.invoke(formatted_messages)
            return response.content
            
        except Exception as e:
            return f"An internal error occurred during generation: {str(e)}"
