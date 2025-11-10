from typing import List, Optional
from src.state.rag_state import RagState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGnodes:
    """Contains node functions for the RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RagState) -> RagState:
        """Retrieve relevant documents using retriever"""
        docs = self.retriever.invoke(state.question)
        return RagState(question=state.question, retrieved_docs=docs)

    def _build_tools(self) -> List[Tool]:
        """Build retriever + Wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {})
                title = meta.get("title") or meta.get("source") or f"doc{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from vectorstore relevant to the query.",
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge or factual queries.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """Create a ReAct agent with the retriever and Wikipedia tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided documents; "
            "use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RagState) -> RagState:
        """Generate an answer using the ReAct agent with retriever + Wikipedia"""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        answer: Optional[str] = None
        if isinstance(result, str):
            answer = result
        else:
            messages = result.get("messages", [])
            if messages:
                answer_msg = messages[-1]
                answer = getattr(answer_msg, "content", None)

        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate an answer.",
        )
