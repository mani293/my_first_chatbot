import streamlit as st
import pdfplumber
import langchain_text_splitters
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.header ("Hello, welcome to the RAG chatbot !")
with st.sidebar:
    st.title("Please upload a document to get started.")
    file = st.file_uploader("Upload document", type="pdf")

#extract content and chunk it
if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    # st.write(text)

    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], 
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings = model.encode(chunks)
    # st.write(embeddings)

    vector_store =FAISS.from_texts(chunks, embeddings)


    user_question =st.text_input("Ask a question about the document:")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = OllamaLLM(model="llama3.2")

    prompt =ChatPromptTemplate.from_template(""" 
        input_variables=["context", "question"], 
        template="You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"  
        """)
    strOutputParser = StrOutputParser()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | strOutputParser
    )

    response = chain.invoke(user_question)
    st.write(response)