# AI Resume Analyzer & Document Query Engine

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=for-the-badge&logo=fastapi)
![RAG](https://img.shields.io/badge/RAG-Advanced-FF6B6B?style=for-the-badge)

📂 [GitHub Repository](https://github.com/Arindam80/Mini-Rag.git) | 👨‍💻 [My LinkedIn Profile](https://www.linkedin.com/in/arindam-mondal-305bb725b/) | 📄 [Resume Link](https://drive.google.com/file/d/18ufWn8vtKXpBXkmBFhTiHttmIBW_njOM/view?usp=drive_link)

<div align="center">
  <a 
    href="https://mini-rag-five.vercel.app/" 
    target="_blank" 
    rel="noopener noreferrer" 
    style="
      display: inline-block;
      padding: 18px 36px;
      font-size: 22px;
      font-weight: bold;
      color: #ffffff;
      background: linear-gradient(45deg, #7e57c2, #512da8);
      text-decoration: none;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      transition: all 0.3s ease-in-out;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    "
    onmouseover="
      this.style.transform='translateY(-3px) scale(1.05)';
      this.style.boxShadow='0 8px 25px rgba(0, 0, 0, 0.4), 0 0 10px #c792ea';
      this.style.background='linear-gradient(45deg, #9575cd, #673ab7)';
    "
    onmouseout="
      this.style.transform='translateY(0) scale(1)';
      this.style.boxShadow='0 4px 15px rgba(0, 0, 0, 0.2)';
      this.style.background='linear-gradient(45deg, #7e57c2, #512da8)';
    "
  >
    🚀 Live Demo
  </a>
</div>
## 📋 Submission Checklist

✅ **Live URL(s):** Application running locally (instructions below)  
✅ **Public GitHub repo:** https://github.com/Arindam80/Mini-Rag.git  
✅ **README with setup, architecture, and resume link:** This document  
✅ **Clear schema/index config:** Detailed in Architecture section  
✅ **Remarks section:** Included below with limitations and future improvements  

## 🚀 Overview

A comprehensive AI-powered Resume Analyzer and Document Query Engine built with a modern RAG (Retrieval-Augmented Generation) pipeline. This application allows users to upload resumes or paste document text, then ask intelligent questions to get AI-powered insights with proper source citations.

## ✨ Features

- 📄 **Dynamic Document Ingestion**: Accepts any block of text or resume upload for real-time querying
- 📋 **Resume Upload**: Upload your resume (.pdf, .txt, .docx) and ask questions about your skills, experience, or education
- 💡 **Smart Suggestions**: Pre-built analysis questions for resume evaluation
- 🤖 **Advanced RAG Pipeline**: Implements state-of-the-art Retrieve-Rerank-Generate workflow
- ✅ **Cited & Grounded Answers**: LLM answers only from provided context, reducing hallucinations
- 🎨 **Modern & Responsive UI**: Dark-themed frontend built with HTML, CSS, and JavaScript
- 📚 **Source References**: See which parts of your document informed the AI's answers
- 🌐 **Separated Deployment**: Frontend on Vercel, backend on Render for optimal performance

## 🏗️ Architecture & Tech Stack

The application follows a decoupled architecture with specialized components:

### Data Flow:
```
Frontend → Backend API → RAG Pipeline → [Qdrant → Cohere → Groq] → Response
```

### Key Components:

| Component | Technology/Provider | Purpose |
|-----------|-------------------|---------|
| Frontend | HTML5, CSS3, Vanilla JavaScript | Responsive User Interface |
| Backend | Python 3.11, FastAPI | API Logic & RAG Orchestration |
| Vector Database | Qdrant Cloud | Storing Text Embeddings |
| Embeddings | Cohere (embed-english-v3.0) | Converting Text to Vectors |
| Reranker | Cohere (rerank-english-v3.0) | Improving retrieval relevance |
| LLM | Groq (Llama 3.1 llama-3.1-8b-instant) | Generating final answers |
| Orchestration | LangChain | Connecting AI components |

### Index Configuration

- **Vector Database:** Qdrant Cloud
- **Embedding Model:** Cohere embed-english-v3.0
- **Vector Dimensionality:** 1024
- **Collection Name:** my_rag_collection_cohere
- **Chunking Strategy:** Recursive with 1000-character chunks, 150-character overlap

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- Git
- API keys for:
  - Qdrant Cloud
  - Cohere
  - Groq

### 1. Clone the Repository

```bash
git clone https://github.com/Arindam80/Mini-Rag.git
cd Mini-Rag
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### 3. Configure Environment Variables

Edit the `.env` file with your API keys:

```env
QDRANT_URL="your_qdrant_cluster_url"
QDRANT_API_KEY="your_qdrant_api_key"
COHERE_API_KEY="your_cohere_api_key"
GROQ_API_KEY="your_groq_api_key"
```

### 4. Run the Backend Server

```bash
uvicorn main:app --reload --port 8000
```

### 5. Run the Frontend

Open the `index.html` file in your browser using a local server (VS Code Live Server recommended).

## 📊 Usage

1. **Upload Content:** Either paste text or upload a resume file
2. **Process:** Click "Process & Embed Document" or "Analyze Resume"
3. **Ask Questions:** Use the query box or click on pre-built suggestions
4. **Review Results:** See AI-generated answers with source citations

### Pre-built Resume Analysis Questions:

- 💪 Strengths & Weaknesses analysis
- 🛠️ Skills identification and evaluation
- 🤖 ATS optimization tips
- 🎯 Job role matching suggestions
- ⭐ Comprehensive resume rating
- 🔍 Missing elements analysis

## 🎯 API Endpoints

- `GET /` - Health check
- `POST /upload` - Process text documents
- `POST /upload-resume` - Process resume files
- `POST /query` - Ask questions about uploaded content
- `GET /health` - Detailed system health check

## 🔧 Technical Implementation

### Document Processing

The system handles multiple file formats:

- **PDF:** Using PyPDF2 for text extraction
- **DOCX:** Using python-docx library
- **DOC:** Using mammoth for legacy Word documents
- **TXT:** Direct text processing with UTF-8 and Latin-1 encoding support

### RAG Pipeline

1. **Chunking:** Recursive text splitting with semantic preservation
2. **Embedding:** Cohere's embed-english-v3.0 model
3. **Storage:** Qdrant vector database with metadata tracking
4. **Retrieval:** Similarity search with configurable parameters
5. **Reranking:** Cohere's rerank-english-v3.0 for precision
6. **Generation:** Groq's Llama 3.1 for answer synthesis

## ✅ Evaluation Results

### Test Case 1: Resume Query

**Context Provided:** User's uploaded resume containing:
- Languages: C, C++, C-Sharp, Python, Java, HTML, CSS, SQL
- Tools: VS Code, Android Studio, Git, GitHub, Visual Studio, Colab, MySQL, Cisco Packet Tracer
- Frameworks/Libraries: Bootstrap, NumPy, Pandas, Matplotlib, .NET
- Operating Systems: Linux (Ubuntu, Kali), Windows

**Question:** "What are my skills?"

**✅ Result:**
> "My skills include programming languages such as C, C++, C-Sharp, Python, Java, HTML, CSS, and SQL; tools like VS Code, Android Studio, Git, GitHub, Visual Studio, Colab, MySQL, and Cisco Packet Tracer; frameworks and libraries including Bootstrap, NumPy, Pandas, Matplotlib, and .NET; and operating systems like Linux (Ubuntu, Kali) and Windows. [1]"

**Analysis:** Correctly extracted all skills from the uploaded resume with proper source citation.

## 📝 Remarks & Considerations

### Limitations

- **File Size:** Maximum 10MB file upload limit
- **Text Length:** Very long documents may have processing delays
- **API Dependencies:** Relies on external services (Qdrant, Cohere, Groq)
- **Cost Considerations:** API usage may incur costs with high volume
- **Format Limitations:** Complex PDF formatting may not extract perfectly

### Trade-offs

- **Speed vs. Accuracy:** Using Groq for faster inference but potentially less nuanced than larger models
- **Simplicity vs. Features:** Focused core functionality rather than extensive enterprise features
- **Local vs. Cloud:** Cloud-based vector database for scalability but requires internet connection

### Future Enhancements

- **Multi-document support:** Compare multiple resumes or documents
- **Export functionality:** Download analysis reports as PDF
- **Custom chunking strategies:** Adaptive chunking based on content type
- **Batch processing:** Handle multiple files simultaneously
- **Advanced filtering:** Metadata-based document filtering
- **Local LLM option:** Support for offline model inference
- **User authentication:** Personalized document collections
- **API rate limiting:** Better management of request volumes
- **Enhanced file parsing:** Better handling of complex document layouts
- **Real-time collaboration:** Multiple users analyzing the same document

## 📞 Support

For questions or issues regarding this implementation, please contact through:

- **GitHub Issues:** [Repository Issues](https://github.com/Arindam80/Mini-Rag/issues)
- **LinkedIn:** [Arindam Mondal](https://www.linkedin.com/messaging/compose/?recipient=arindam-mondal-305bb725b)

---

