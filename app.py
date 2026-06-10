import streamlit as st
import os
from dotenv import load_dotenv

# Groq & AI Tools
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key from .env
load_dotenv()

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="Lexi AI | Business Law", 
    page_icon="⚖️", 
    layout="wide"
)

# Custom CSS for a clean, professional "ChatGPT/Gemini" look
st.markdown("""
    <style>
    /* 1. Reset text behavior so it adapts to Light/Dark phone themes */
    .stMarkdown p, .stMarkdown li, span {
        color: inherit !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    /* 2. Style the chat containers for better spacing on small screens */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 12px !important;
        padding: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }

    /* 3. Make headings clean and sharp on mobile devices */
    h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        padding-bottom: 5px !important;
    }
    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-top: 10px !important;
    }

    /* 4. Fix padding issues for mobile browsers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
# --- 2. THE BRAIN SETUP (RAG) ---
@st.cache_resource
def load_knowledge_base():
    # Scans the 'laws' folder for your PDFs
    loader = PyPDFDirectoryLoader("laws")
    docs = loader.load()
    
    # Break the law books into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(docs)
    
    # Create the search engine using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1048/1048953.png", width=80)
    st.title("Lexi Settings")
    st.divider()
    st.markdown("**Knowledge Base:** 5 Law Acts Loaded ✅")
    st.markdown("**Model:** Llama 3.3 (High Speed)")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN INTERFACE ---
st.title("⚖️ Lexi: Business Law Assistant")
st.caption("Advanced Legal RAG System for University Students")

# Quick start buttons for the user
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Explain elements of a contract"):
        st.session_state.temp_prompt = "What are the essential elements of a valid contract?"
with col2:
    if st.button("How to form a Private Ltd. Company?"):
        st.session_state.temp_prompt = "What is the process to form a Private Limited Company?"
with col3:
    if st.button("Rights of a partner in a firm"):
        st.session_state.temp_prompt = "Explain the rights of a partner in a firm according to the Partnership Act."

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. CHAT LOGIC ---
# Handle prompt from input box OR quick buttons
prompt = st.chat_input("Ask Lexi a legal question...")
if hasattr(st.session_state, 'temp_prompt'):
    prompt = st.session_state.temp_prompt
    del st.session_state.temp_prompt

if prompt:
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ GROQ_API_KEY missing! Please check your .env file.")
        else:
            with st.spinner("Lexi is analyzing legal documents..."):
                try:
                    # Initialize the Brain
                    vector_store = load_knowledge_base()
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
                    
                    # THE SYSTEM PROMPT: This controls the "clean" visual output
                    system_prompt = (
                        "You are Lexi, a professional legal assistant. Use the provided context to answer. "
                        "FORMATTING RULES:\n"
                        "1. Use '###' for clear section headers.\n"
                        "2. Use double line breaks between ALL paragraphs for white space.\n"
                        "3. Use bold text for key terms and Act names.\n"
                        "4. Always cite the specific Act and Section if found in the context.\n"
                        "5. If you don't know, suggest checking with a legal professional.\n\n"
                        "Context: {context}"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # RAG Chain Setup
                    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                    qa_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, qa_chain)
                    
                    # Get Answer
                    response = rag_chain.invoke({"input": prompt})
                    full_response = response["answer"]
                    
                    # Display Answer
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")