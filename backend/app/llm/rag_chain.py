from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .llm_provider import get_llm
from .retriever import RehabDbRetriever
from langchain.schema.runnable import RunnableLambda

def get_rag_chain():
    """
    Constructs and returns a complete RAG chain for the rehab assistant.
    """
    # ... (template and prompt definition remains the same) ...
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

    # 💡 FIX: Pass the user_id via the input dict and handle it in a RunnableLambda 
    # The chain now takes {user_id: str, question: str} as input
    chain = (
        {
            # 💡 FIX: Use RunnableLambda to dynamically create the retriever
            "context": RunnableLambda(lambda x: RehabDbRetriever(user_id=x['user_id']).invoke(x['question'])),
            "question": RunnablePassthrough(),
            "user_id": RunnablePassthrough() # Pass user_id through to be used in context generator
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
    We now pass user_id and question explicitly.
    """
    # 💡 FIX: Pass question and user_id as a dictionary input
    return rag_chain.invoke({'question': question, 'user_id': user_id})
