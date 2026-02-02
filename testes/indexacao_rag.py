from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

#Para um arquivo específico
def extract_text_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    doc = loader.load()
    content = "\n".join([page.page_content for page in doc])
    return content

#Para vários arquivos 
def extract_all_text_pdfs(docs_path):
    docs_path = Path(docs_path)
    pdf_files = [f for f in docs_path.glob("*.pdf")]
    loaded_contents = [extract_text_pdf(pdf) for pdf in pdf_files]
    return loaded_contents

#Quebra o texto em pedaços
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

#Embedding de dados
def embedding(chunk):
    #embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = "BAAI/bge-M3"

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    result_embedding = embeddings.embed_query(chunk)
    return result_embedding
    
def inicio():
    text_pdfs = extract_all_text_pdfs("./base")

    chunks = split_text(text_pdfs)

    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("index_faiss")

    #Carrega dados
    #db = FAISS.load_local("index_faiss", embeddings)

    #Recupera dados
    retriever = vectorstore.as_retriever(searchType='simalarity', search_kwargs={"k": 6})
    return retriever


