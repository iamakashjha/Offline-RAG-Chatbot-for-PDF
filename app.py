import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain



def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def create_faiss_vector_store(text, path='faiss_index'):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)


def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def build_qa_chain(vector_store_path='faiss_index'):
    vector_store = load_faiss_vector_store(vector_store_path)
    retriever = vector_store.as_retriever()

    llm = Ollama(model="llama3.2")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    return qa_chain


try:
    st.title("Offline RAG Chatbot with FAISS & Ollama")
    st.write("Upload a PDF and ask questions about it!")

    upload_file = st.file_uploader("Upload your PDF", type="pdf")

    if upload_file is not None:
        os.makedirs("uploaded", exist_ok=True)
        pdf_path = os.path.join("uploaded", upload_file.name)

        with open(pdf_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        text = extract_text_from_pdf(pdf_path)

        st.info("Creating FAISS vector store...")
        create_faiss_vector_store(text)

        st.info("Initializing chatbot...")
        qa_chain = build_qa_chain()
        st.success("Chatbot is ready!")

    if 'qa_chain' in locals():
        question = st.text_input("Ask a question about the uploaded PDF:")
        if question:
            st.info("Querying the document...")
            answer = qa_chain.run(question)
            st.success(f"Answer: {answer}")

except Exception as e:
    st.error(f"Error: {e}")
