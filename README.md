# RAG-Document-Search

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to search information or queries about upload documents, perform semantic search, and ask questions over them using LLM-powered responses.
This project demonstrates how retrieval and generation can be combined to create a powerful document understanding system.

---

🌐 Deployment Link

👉 [Live Demo](https://sahana1234n-rag-document-search-app-pxttna.streamlit.app/)


---

🚀 Features

- 📂 Document Ingestion — Upload PDFs, text, or docx files for processing

- 🧮 Text Chunking & Embeddings — Converts text into embeddings using vector representations

- 🧠 Vector Database — Stores and retrieves semantically relevant chunks

- 💬 RAG Pipeline — Combines retrieved context with LLMs to generate accurate answers

- ⚡ Streamlit Interface — Simple and interactive web app for querying documents

- ☁️ Deployed on  Streamlit Cloud

---

🏗️ Project Architecture
Start → Document Upload → Text Split & Embeddings → Store in Vector DB → Query → Retrieve Relevant Chunks → LLM → Response

---

🔧 Modules

| Module                  | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `document_processor.py` | Extracts and cleans text from uploaded documents              |
| `vectorstore.py`        | Creates and manages vector database using FAISS               |
| `graph_builder.py`      | Builds retrieval graph for contextual mapping                 |
| `config.py`             | Handles API keys and configuration                            |
| `app.py`                | Streamlit frontend for user interaction                       |

---

🧩 Tech Stack

- Frontend: Streamlit
  
- Backend: Python

- Libraries: LangChain, FAISS, HuggingFace Transformers, PyPDF2 , LangGraph

- LLM: GROQ model

- Deployment: Streamlit Cloud

---

⚙️ Installation
# Clone this repository
- git clone https://github.com/<your-username>/RAG-Document-Search.git
- cd Rag-Document-Search

# Create virtual environment
- python -m venv venv
- venv\Scripts\activate on Windows

# Install dependencies
- pip install -r requirements.txt

▶️ Run the App
- streamlit run app.py

---

📊 Example Use Case

Upload a research paper or company policy document and ask questions like:

- “What are the key findings in this paper?”
- “What does section 3 say about privacy policy?”

The system retrieves the most relevant sections and respond back with LLM-generated answers.

---

🏆 Future Improvements

- Add multi-document retrieval

- Integrate advanced LLMs (Gemini / Llama 3)

- Implement chat history memory

- Improve UI/UX design

---

👩‍💻 Author

Sahana K N
📧 sahana86gowda@gmail.com
🌐 www.linkedin.com/in/sahanakn2002




