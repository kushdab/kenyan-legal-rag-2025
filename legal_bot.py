import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Page Configuration
st.set_page_config(page_title="Kenyan Legal RAG 2025", layout="wide")
st.title("⚖️ Kenyan High Court Legal Assistant")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("Upload Kenyan High Court Judgments (PDF)", accept_multiple_files=True)
    process_button = st.button("Index Documents")

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_documents(files):
    documents = []
    for file in files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())
        os.remove(temp_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore

if process_button and openai_api_key:
    if uploaded_files:
        with st.spinner("Indexing legal precedents..."):
            st.session_state.vectorstore = process_documents(uploaded_files)
            st.success("Database ready for legal queries!")
    else:
        st.error("Please upload PDF documents.")

# Chat Interface
st.divider()
if st.session_state.vectorstore:
    # System prompt to enforce Kenyan context
    system_template = """
    You are an expert Kenyan Legal Assistant. Use the provided Kenyan High Court judgments 
    and the Constitution of Kenya 2010 to answer queries. 
    If the answer is not in the context, state that you cannot find a direct precedent. 
    Always cite the specific Case Name and Petition Number if available.
    """
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=openai_api_key, temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about Kenyan precedents (e.g., land disputes, constitutional petitions)"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = qa_chain.invoke({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Please enter your OpenAI API key and upload legal documents in the sidebar to start.")