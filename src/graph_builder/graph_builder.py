"""Graph builder for langGraph workflow"""

from langgraph.graph import StateGraph , END
from src.state.rag_state import RagState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self , retriever , llm):
        """
        Initializes graph builder
        
        Args:
           retreiver: Document retriever instance
           llm: Language model instance
        """

        self.nodes = RAGNodes(retriever , llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        #create stategraph
        builder = StateGraph(RagState)

        #add nodes
        builder.add_node("retriever" , self.nodes.retrieve_docs)
        builder.add_node("responder" , self.nodes.generate_answer)

        # set entry point
        builder.set_entry_point("retriever")

        #Add edges
        builder.add_edge("retriever" , "responder")
        builder.add_edge("responder" , END)

        #compile graph
        self.graph = builder.compile()
        return self.graph           

    def run(self , question:str)->dict:
        """
        Run the RAG workflow
        
        Args:
            question: user question
            
        Returns:
            Final state with answer 
        """
        if self.graph is None:
            self.build()

        initial_state = RagState(question=question)
        return self.graph.invoke(initial_state)         
    