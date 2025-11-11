import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder



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
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

# Cached RAG Initialization

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached for performance)."""
    try:
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
    """Main application."""
    init_session_state()

    # Title and description
    st.title("🔎 RAG Document Search")
    st.markdown("Ask questions about the loaded documents below. \n 1. Attention is all you need \n 2. Langchain \n 3. RAG ")

    # Initialize RAG system
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing the RAG system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # Search Form
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder=""
        )
        submit = st.form_submit_button("🔍 Search")

    # Process Search Query
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching... please wait ⏳"):
                start_time = time.time()

                # Run the query through RAG system
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time

                # Append to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result.get('answer', 'No answer generated.'),
                    'time': elapsed_time
                })

                # Display the answer
                st.subheader("✅ Answer")
                st.success(result.get('answer', 'No answer generated.'))

                # Show retrieved docs in expander
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


# Run the App

if __name__ == "__main__":
    main()
