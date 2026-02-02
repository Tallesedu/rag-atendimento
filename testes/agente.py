from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from indexacao_rag import inicio
import os
import getpass

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)


def load_llm():
    llm = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=None,
        max_retries=2,
        timeout=None
    )

    return llm

template_rag = """
    Context: {context}
    Prompt: {input}
"""
prompt_rag = PromptTemplate(
    input_variables=['context', 'input'],
    template=template_rag
)

llm = load_llm()

retriever = inicio()

chain_rag = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

res = chain_rag.invoke("Quais os veículos você tem disponível?")
print(res)