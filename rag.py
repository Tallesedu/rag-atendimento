import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain.messages import AIMessage, HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


### Carregamento da LLM
def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )
  return llm

llm = load_llm('llama-3.3-70b-versatile', 0.7)

### Exibição do resultado
def show_res(res):
    from IPython.display import Markdown
    if "</think>" in res:
        res = res.split("</think>")[-1].strip()
    else:
        res = res.strip()  # fallback se não houver tag

    display(Markdown(res))


### Extração do conteúdo
def extract_text_pdf(file_path):
  loader = PyMuPDFLoader(file_path)
  doc = loader.load()
  content = "\n".join([page.page_content for page in doc])
  return content


### Indexação e recuperação
def config_retriever(folder_path="./base"):
    # Carregar documentos
    docs_path = Path(folder_path)
    pdf_files = [f for f in docs_path.glob("*.pdf")]

    if len(pdf_files) < 1:
        st.error("Nenhum arquivo PDF carregado")
        st.stop()

    loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]

    # Divisão em pedaços de texto / Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = []
    for doc in loaded_documents:
        chunks.extend(text_splitter.split_text(doc))

    # Embeddings
    embedding_model = "BAAI/bge-m3" #sentence-transformers/all-mpnet-base-v2

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Armazenamento
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    vectorstore.save_local('index_faiss')

    # Configurando o recuperador de texto / Retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':4}
    )

    return retriever

### Chain da RAG
def config_rag_chain(llm, retriever):

  # Prompt de contextualização
  context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

  context_q_system_prompt = context_q_system_prompt
  context_q_user_prompt = "Question: {input}"
  context_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", context_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", context_q_user_prompt),
      ]
  )

  # Chain para contextualização
  history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=context_q_prompt
  )

  # Prompt para perguntas e respostas (Q&A)
  system_prompt = """Você é um assistente virtual prestativo e está respondendo perguntas gerais sobre os serviços de uma empresa.
  Use os seguintes pedaços de contexto recuperado para responder à pergunta.
  Se você não sabe a resposta, apenas comente que não sabe dizer com certeza.
  Mas caso seja uma dúvida muito comum, pode sugerir como alternativa uma solução possível.
  Mantenha a resposta concisa.
  Responda em português. \n\n"""

  qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "Pergunta: {input}\n\n Contexto: {context}"),
  ])

  # Configurar LLM e Chain para perguntas e respostas (Q&A)

  qa_chain = create_stuff_documents_chain(llm, qa_prompt)

  rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
  )

  return rag_chain