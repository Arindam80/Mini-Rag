import os
import io
import tempfile
import traceback
from dotenv import load_dotenv
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Document Processing Imports
import PyPDF2
import docx
import mammoth

# LangChain Imports
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Qdrant

# Import specific Cohere error classes
from cohere.errors import (
    BadRequestError, UnauthorizedError, TooManyRequestsError, InternalServerError,
    ServiceUnavailableError, NotFoundError, ForbiddenError, UnprocessableEntityError
)

# --- INITIAL SETUP ---

# Load environment variables from .env file
load_dotenv()

# Create a tuple for catching multiple Cohere errors
COHERE_ERRORS = (
    BadRequestError, UnauthorizedError, TooManyRequestsError, InternalServerError,
    ServiceUnavailableError, NotFoundError, ForbiddenError, UnprocessableEntityError
)

# Instantiate the FastAPI app
app = FastAPI(
    title="AI Resume Analyzer",
    description="Enhanced RAG system with resume analysis capabilities"
)

# --- CORS MIDDLEWARE ---
# CORRECTED: Removed trailing slash from Vercel URL for better matching.
origins = [
    "https://mini-rag-five.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES & CONSTANTS ---
QDRANT_COLLECTION_NAME = "my_rag_collection_cohere"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}

# --- PYDANTIC MODELS ---
class UploadData(BaseModel):
    text: str

class QueryData(BaseModel):
    question: str

# --- DOCUMENT PROCESSING FUNCTIONS ---

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = "".join(paragraph.text + "\n" for paragraph in doc.paragraphs)
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def extract_text_from_doc(file_content: bytes) -> str:
    """Extract text from DOC file using mammoth."""
    try:
        result = mammoth.extract_raw_text(io.BytesIO(file_content))
        return result.value.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading DOC: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file."""
    try:
        return file_content.decode('utf-8').strip()
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1').strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading TXT file: {str(e)}")

def process_file(file_content: bytes, filename: str) -> str:
    """Process uploaded file and extract text based on file extension."""
    file_ext = os.path.splitext(filename)[1].lower()
    handlers = {
        '.pdf': extract_text_from_pdf,
        '.docx': extract_text_from_docx,
        '.doc': extract_text_from_doc,
        '.txt': extract_text_from_txt,
    }
    if handler := handlers.get(file_ext):
        return handler(file_content)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

def chunk_and_embed_text(text: str) -> int:
    """Chunk text and store embeddings in Qdrant."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        documents = [
            Document(page_content=chunk, metadata={"position": i + 1})
            for i, chunk in enumerate(chunks)
        ]
        
        embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-english-v3.0"
        )
        
        # Using QdrantVectorStore creates the collection and adds documents
        QdrantVectorStore.from_documents(
            documents,
            embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=QDRANT_COLLECTION_NAME,
            force_recreate=True, # Deletes and recreates the collection each time
        )
        
        return len(documents)
    except COHERE_ERRORS as e:
        raise HTTPException(status_code=500, detail=f"Cohere API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "AI Resume Analyzer API is running", "version": "2.1"}

@app.post("/upload")
async def upload_text(data: UploadData):
    """Endpoint for uploading, chunking, and embedding raw text."""
    try:
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        chunk_count = chunk_and_embed_text(data.text)
        return {"message": f"Successfully processed and uploaded {chunk_count} chunks", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Endpoint for uploading resume files (PDF, DOC, DOCX, TXT)."""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        
        file_content = await file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        extracted_text = process_file(file_content, file.filename)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in the file")
        
        chunk_count = chunk_and_embed_text(extracted_text)
        return {"message": f"Successfully analyzed '{file.filename}' and created {chunk_count} chunks", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume processing failed: {str(e)}")

@app.post("/query")
async def query(data: QueryData):
    """Endpoint for querying the RAG pipeline."""
    try:
        if not data.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Initialize components
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        
        # IMPROVED: Check if the collection exists and has points before querying
        try:
            collection_info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            if collection_info.points_count == 0:
                raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a document first.")
        except Exception:
             raise HTTPException(status_code=404, detail="No document collection found. Please upload a document first.")

        embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0")
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)
        reranker = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"), top_n=5)
        
        vector_store = Qdrant(client=client, collection_name=QDRANT_COLLECTION_NAME, embeddings=embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        retrieved_docs = retriever.invoke(data.question)
        reranked_docs = reranker.compress_documents(documents=retrieved_docs, query=data.question)
        
        if not reranked_docs:
            return {"answer": "I could not find any relevant information in the document to answer your question.", "sources": []}

        context_text = "\n\n".join([f"Source [{doc.metadata['position']}]:\n{doc.page_content}" for doc in reranked_docs])
        
        # IMPROVED: A more robust prompt template for better, grounded answers
        prompt_template = """
        You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided context.
        Follow these rules strictly:
        1.  Analyze the provided "Context" section, which contains numbered sources.
        2.  Formulate a clear and concise answer to the "Question".
        3.  Base your entire answer on the information found within the context. Do not use any external knowledge.
        4.  Cite the sources you used to construct your answer by adding the source number in brackets, like `[1]`, `[2]`, etc.
        5.  If the answer cannot be found in the provided context, you must explicitly state: "I could not find an answer to this question in the provided document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        response_llm = chain.invoke({"context": context_text, "question": data.question})
        
        source_documents = [{"content": doc.page_content, "position": doc.metadata['position']} for doc in reranked_docs]
        
        return {"answer": response_llm.content, "sources": source_documents, "status": "success"}

    except COHERE_ERRORS as e:
        raise HTTPException(status_code=500, detail=f"Cohere API error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health")
def health_check():
    """Detailed health check for environment variables and services."""
    try:
        required_vars = ["COHERE_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        return {"status": "healthy", "services": "all variables present"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
