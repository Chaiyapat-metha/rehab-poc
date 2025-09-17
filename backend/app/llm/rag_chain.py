from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .llm_provider import get_llm
from .retriever import RehabDbRetriever

def get_rag_chain():
    """
    Constructs and returns a complete RAG chain for the rehab assistant.
    """
    template = """
    You are a helpful and encouraging AI physical therapy assistant.
    Your language is Thai.
    Answer the user's question based only on the following context.
    If the context is empty, say you don't have enough information.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()

    # The chain is defined using LangChain Expression Language (LCEL)
    chain = (
        {
            "context": RehabDbRetriever(user_id="placeholder"), # Retriever is part of the chain
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Create a single instance of the chain when the module is loaded
rag_chain = get_rag_chain()

def invoke_rag_chain(user_id: str, question: str) -> str:
    """
    Invokes the RAG chain for a specific user and question.
    We dynamically update the user_id in the retriever for each call.
    """
    # This is how you update a part of the chain dynamically
    chain_with_user = rag_chain.assign(
        context=RehabDbRetriever(user_id=user_id)
    )
    return chain_with_user.invoke(question)
