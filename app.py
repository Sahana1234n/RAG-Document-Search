import streamlit as st
from pathlib import Path
import sys
import time
import os
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Load GROQ API key from Streamlit secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 GROQ_API_KEY not found in Streamlit secrets!")
    GROQ_API_KEY = None

# Pass the key to Config
Config.GROQ_API_KEY = GROQ_API_KEY

# Set a custom USER_AGENT for HTTP requests
os.environ["USER_AGENT"] = "MyRAGApp/1.0"

# Add src to path
sys.path.append(str(Path(__file__).parent))


# Streamlit Page Configuration
st.set_page_config(
    page_title="RAG Document Search",
    layout="centered",
    page_icon="🔍"
)

# Simple CSS Styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)


# Session State Initialization
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


# Cached RAG Initialization
@st.cache_resource
def initialize_rag():
    try:
        if not Config.GROQ_API_KEY:
            st.error("Cannot initialize RAG system: API key missing.")
            return None, 0

        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        vector_store = VectorStore()
        urls = Config.DEFAULT_URLS  # Use default URLs

        # Process documents
        documents = doc_processor.process(urls)

        # Create vector store
        vector_store.create_vectorstore(documents)

        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"⚠️ Failed to initialize: {str(e)}")
        return None, 0


# Main App
def main():
    init_session_state()

    # Title
    st.title("🔎 RAG Document Search")
    st.markdown("Ask questions about the loaded documents below. \n1. Attention is all you need\n2. Langchain\n3. RAG")

    # Search Form
    with st.form("search_form"):
        question = st.text_input("Enter your question:", placeholder="")
        submit = st.form_submit_button("🔍 Search")

    if submit and question:
        # Initialize RAG system if not yet initialized
        if not st.session_state.initialized:
            if GROQ_API_KEY:
                with st.spinner("🚀 Initializing the RAG system..."):
                    rag_system, num_chunks = initialize_rag()
                    if rag_system:
                        st.session_state.rag_system = rag_system
                        st.session_state.initialized = True
                        st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
            else:
                st.error("🚨 GROQ_API_KEY missing. Cannot initialize RAG system.")

        # Run query if RAG system is ready
        if st.session_state.rag_system:
            with st.spinner("Searching... please wait ⏳"):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time

                # Append to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result.get('answer', 'No answer generated.'),
                    'time': elapsed_time
                })

                # Display answer
                st.subheader("✅ Answer")
                st.success(result.get('answer', 'No answer generated.'))

                # Show retrieved docs
                with st.expander("📚 Source Documents"):
                    docs = result.get('retrieved_docs', [])
                    if docs:
                        for i, doc in enumerate(docs[:5], 1):
                            st.text_area(
                                f"Document {i}",
                                doc.page_content[:500] + "...",
                                height=120,
                                disabled=True
                            )
                    else:
                        st.info("No source documents found.")

                # Show response time
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Display Search History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("🕒 Recent Searches")
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                st.caption(f"🕐 {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()
