"""LangGraph nodes for RAG workflow"""

from src.state.rag_state import RagState


class RAGNodes:

    def __init__(self , retriever , llm):
        """
        Initializes RAG Nodes 
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self , state:RagState)->RagState:
        """
        Retrieve relevant documents

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved documents

        """    
        docs = self.retriever.invoke(state.question)
        return RagState(
            question = state.question , 
            retrieved_docs= docs
        )
    
    def generate_answer(self , state:RagState)->RagState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retreievd documents
            
        Returns:
            Updated RAG state with generated answer  
         """
        #combine retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # create prompt
        prompt =  f"""
        You are a knowledgeable AI assistant. 
        Use the retrieved documents provided in the context below to answer the user question clearly and directly. 
        If the context provides related information, summarize it naturally. 
        Avoid saying phrases like "the context does not mention". 
        If the answer truly cannot be found, say "I could not find that in the available documents." 
        Be concise and factual.
        
        context : 
        {context} 

        Question:
        {state.question}"""

        #Generate response
        response = self.llm.invoke(prompt)

        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
