import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from rag import config_retriever, config_rag_chain, load_llm
load_dotenv()

st.set_page_config(page_title="Atendimento 🤖", page_icon="🤖")
st.title("Atendimento")

llm = load_llm('llama-3.3-70b-versatile', 0.7)
path = "./base"

### Interação com chat
def chat_llm(rag_chain, input):

  st.session_state.chat_history.append(HumanMessage(content=input))

  response = rag_chain.invoke({
      "input": input,
      "chat_history": st.session_state.chat_history
  })

  res = response["answer"]
  res = res.split("</think>")[-1].strip() if "</think>" in res else res.strip()

  st.session_state.chat_history.append(AIMessage(content=res))

  return res

input = st.chat_input("Digite sua mensagem aqui...")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
      AIMessage(content = "Olá, sou o seu assistente virtual! Como posso te ajudar?"),
  ]

if "retriever" not in st.session_state:
  st.session_state.retriever = None

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)

if input is not None:
  with st.chat_message("Human"):
    st.markdown(input)

  with st.chat_message("AI"):
    if st.session_state.retriever is None:
      st.session_state.retriever = config_retriever(path)
    rag_chain = config_rag_chain(llm, st.session_state.retriever)
    res = chat_llm(rag_chain, input)
    st.write(res)