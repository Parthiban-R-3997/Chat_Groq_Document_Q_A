import streamlit as st
import os
import tempfile
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()




st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

st.title("Chat Groq Document Q&A")

# Custom prompt template
custom_context_input = """
<context>
{context}
<context>
Questions:{input}
"""

# Default prompt template
default_prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

def vector_embedding(pdf_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents = []
    for pdf_file in pdf_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the PDF from the temporary file path
        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load()) ## append the files

        # Remove the temporary file
        os.remove(tmp_file_path)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    st.success("Document embedding is completed!")


# Define model options
model_options = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Sidebar elements
with st.sidebar:
    st.header("Configuration")
    st.markdown("Enter your API keys below:")
    groq_api_key = st.text_input("Enter your GROQ API Key", type="password", help="Get your API key from [GROQ Console](https://console.groq.com/keys)")
    google_api_key = st.text_input("Enter your Google API Key", type="password", help="Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
    langsmith_api_key = st.text_input("Enter your Langsmith API Key", type="password",placeholder="For Tracing the flows (Optional!)", help="Get your API key from [Langsmith Console](https://smith.langchain.com/o/2a79134f-7562-5c92-a437-96b080547a1e/settings)")
    selected_model = st.selectbox("Select any Groq Model", model_options)
    os.environ["GOOGLE_API_KEY"]=str(google_api_key)
    os.environ["LANGCHAIN_API_KEY"]=str(langsmith_api_key)
    # Langmith tracking
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    st.markdown("Upload your PDF files:")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")


    # Custom prompt text areas
    custom_prompt_template = st.text_area("Custom Prompt Template", placeholder="Enter your custom prompt here to set the tone of the message...(Optional)")

    if st.button("Start Document Embedding"):
        if uploaded_files:
            vector_embedding(uploaded_files)
            st.success("Vector Store DB is Ready")
        else:
            st.warning("Please upload at least one PDF file.")

# Main section for question input and results
prompt1 = st.text_area("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    if custom_prompt_template:
        custom_prompt = custom_prompt_template + custom_context_input
        prompt = ChatPromptTemplate.from_template(custom_prompt)
    else:
        prompt = ChatPromptTemplate.from_template(default_prompt_template)
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------") 
