import streamlit as st
import os
from dotenv import load_dotenv
from prompts import LEXI_SYSTEM_PROMPT

# Groq & LangChain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lexi | Business Law AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME — Design A: White Marble ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300&family=Outfit:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #F7F5F0 !important;
    color: #2a2825 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 0.5px solid #EAE7E0 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #5a564f !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #F0EDE6 !important;
    margin: 14px 0 !important;
}

/* ── Main container ── */
.main .block-container {
    background-color: #F7F5F0 !important;
    padding: 1.5rem 2.5rem 4rem 2.5rem !important;
    max-width: 860px !important;
}

/* ── Page header ── */
.lexi-header {
    padding: 28px 0 20px;
    border-bottom: 0.5px solid #E0DDD6;
    margin-bottom: 24px;
}
.lexi-eyebrow {
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #C9A84C;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 6px;
}
.lexi-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #1A1A2E;
    letter-spacing: 0.3px;
    line-height: 1.15;
    margin: 0 0 6px 0;
}
.lexi-title em {
    font-style: italic;
    color: #C9A84C;
}
.lexi-subtitle {
    font-size: 12px;
    color: #B0AA9E;
    letter-spacing: 0.4px;
    font-family: 'Outfit', sans-serif;
}

/* ── Suggestion chips label ── */
.lexi-chips-label {
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #C5BFB5;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 12px;
    margin-top: 4px;
}

/* ── Suggestion chip buttons ── */
.stButton > button {
    background-color: #FFFFFF !important;
    border: 0.5px solid #DDD9D2 !important;
    border-radius: 20px !important;
    color: #7a746b !important;
    font-size: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    padding: 7px 16px !important;
    transition: all 0.18s ease !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
    width: 100% !important;
    text-align: left !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: #C9A84C !important;
    color: #A07830 !important;
    background-color: #FFFDF7 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Sidebar action button ── */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 7px !important;
    width: 100% !important;
    color: #B0AA9E !important;
    border-color: #EAE7E0 !important;
    font-size: 12px !important;
    text-align: center !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #D4CFC6 !important;
    color: #7a746b !important;
    background-color: #F7F5F0 !important;
    transform: none !important;
}

/* ── Sidebar selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #F7F5F0 !important;
    border: 0.5px solid #E0DDD6 !important;
    border-radius: 8px !important;
    color: #5a564f !important;
    font-size: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 0.5px solid #EAE7E0 !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    padding: 6px 10px !important;
    box-shadow: none !important;
}
[data-testid="stChatMessage"] p {
    color: #2a2825 !important;
    font-size: 14.5px !important;
    line-height: 1.8 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stChatMessage"] strong {
    color: #1A1A2E !important;
    font-weight: 500 !important;
}
[data-testid="stChatMessage"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    color: #1A1A2E !important;
    font-weight: 500 !important;
    margin-top: 12px !important;
    border-bottom: 0.5px solid #F0EDE6 !important;
    padding-bottom: 4px !important;
}

/* User bubble — gold left accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #FFFDF7 !important;
    border-left: 2.5px solid #C9A84C !important;
    border-radius: 0 14px 14px 0 !important;
}

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    border: 0.5px solid #DDD9D2 !important;
    border-radius: 14px !important;
    color: #2a2825 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #C5BFB5 !important;
    font-style: italic !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px #C9A84C22 !important;
}

/* ── Source expander ── */
[data-testid="stExpander"] {
    background: #F7F5F0 !important;
    border: 0.5px solid #E0DDD6 !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #B0AA9E !important;
    font-size: 11px !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #7a746b !important;
}

/* ── Error / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 13px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #E0DDD6; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #C5BFB5; }
</style>
""", unsafe_allow_html=True)


# ── KNOWLEDGE BASE (with FAISS persistence) ──────────────────────────────────
FAISS_PATH = "faiss_index"

@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Load from disk if already indexed — cold start goes from ~30s to <1s
    if os.path.exists(FAISS_PATH):
        vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store

    # First run: load PDFs, chunk, embed, save
    loader = PyPDFDirectoryLoader("laws")
    docs = loader.load()

    # Smaller chunks = better precision for legal section retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_PATH)
    return vector_store


# ── RAG CHAIN (cached — not rebuilt on every message) ────────────────────────
@st.cache_resource(show_spinner=False)
def build_chain(_vector_store):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", LEXI_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    retriever = _vector_store.as_retriever(
        search_type="mmr",          # MMR = diverse chunks, not just top matches
        search_kwargs={"k": 7, "fetch_k": 20}
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo + brand
    st.markdown("""
    <div style='padding: 28px 4px 20px; border-bottom: 0.5px solid #F0EDE6;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='width:42px; height:42px; background:#1A1A2E; border-radius:10px;
                        display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0;'>⚖️</div>
            <div>
                <div style='font-family: Cormorant Garamond, serif; font-size:21px;
                            color:#1A1A2E; font-weight:500; line-height:1;'>Lexi</div>
                <div style='font-size:9px; color:#C9A84C; letter-spacing:2.5px;
                            text-transform:uppercase; margin-top:3px;'>Business Law AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Law acts list
    st.markdown("""
    <p style='font-size:9px; letter-spacing:2.5px; text-transform:uppercase;
              color:#C5BFB5; margin:0 0 10px;'>Knowledge Base</p>
    """, unsafe_allow_html=True)

    acts = [
        ("📜", "Contract Act", "1872"),
        ("🏢", "Companies Act", "1994"),
        ("🤝", "Partnership Act", "1932"),
        ("🛍️", "Sale of Goods Act", "1930"),
        ("🧾", "Negotiable Instruments Act", "1881"),
    ]
    for icon, name, year in acts:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; padding:7px 10px;
                    border:0.5px solid #F0EDE6; border-radius:7px; margin-bottom:5px;
                    background:#FAFAF8;'>
            <span style='font-size:13px;'>{icon}</span>
            <div>
                <span style='font-size:12px; color:#5a564f; font-weight:400;'>{name}</span>
                <span style='font-size:10px; color:#C5BFB5; margin-left:4px;'>{year}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px; border-top:0.5px solid #F0EDE6; margin-top:8px;'></div>", unsafe_allow_html=True)

    # Act filter
    st.markdown("""
    <p style='font-size:9px; letter-spacing:2.5px; text-transform:uppercase;
              color:#C5BFB5; margin:0 0 8px;'>Filter by Act</p>
    """, unsafe_allow_html=True)

    act_filter = st.selectbox(
        label="act_filter",
        options=[
            "All Acts",
            "Contract Act",
            "Companies Act",
            "Partnership Act",
            "Sale of Goods Act",
            "Negotiable Instruments Act",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:16px; border-top:0.5px solid #F0EDE6; margin-top:12px;'></div>", unsafe_allow_html=True)

    # Stats row
    msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.markdown(f"""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px;'>
        <div style='background:#F7F5F0; border-radius:8px; padding:10px 12px;'>
            <div style='font-size:10px; color:#C5BFB5; letter-spacing:0.5px; margin-bottom:2px;'>Model</div>
            <div style='font-size:12px; color:#5a564f; font-weight:500;'>Llama 3.3</div>
        </div>
        <div style='background:#F7F5F0; border-radius:8px; padding:10px 12px;'>
            <div style='font-size:10px; color:#C5BFB5; letter-spacing:0.5px; margin-bottom:2px;'>Questions</div>
            <div style='font-size:12px; color:#5a564f; font-weight:500;'>{msg_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
    <p style='font-size:10px; color:#D4CFC6; text-align:center; margin-top:20px; letter-spacing:0.3px;'>
        Groq · FAISS · LangChain
    </p>
    """, unsafe_allow_html=True)


# ── MAIN HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='lexi-header'>
    <div class='lexi-eyebrow'>Advanced Legal RAG · University Edition</div>
    <div class='lexi-title'>Ask <em>Lexi</em> anything.</div>
    <div class='lexi-subtitle'>Powered by Groq &nbsp;·&nbsp; 5 Bangladeshi Law Acts &nbsp;·&nbsp; Llama 3.3</div>
</div>
""", unsafe_allow_html=True)

# ── QUICK START BUTTONS ──────────────────────────────────────────────────────
st.markdown("<div class='lexi-chips-label'>Try asking</div>", unsafe_allow_html=True)

QUICK_QUESTIONS = {
    "Elements of a valid contract": "What are the essential elements of a valid contract under the Contract Act?",
    "How to form a Private Ltd. company": "What is the process to form a Private Limited Company under the Companies Act?",
    "Rights of a partner in a firm": "What are the rights of a partner in a partnership firm?",
    "What is a negotiable instrument?": "What is a negotiable instrument and what are its characteristics?",
    "Transfer of property in sale": "When does property in goods transfer from seller to buyer under the Sale of Goods Act?",
    "Void vs voidable contract": "What is the difference between a void contract and a voidable contract?",
}

cols = st.columns(3)
for i, (label, full_q) in enumerate(QUICK_QUESTIONS.items()):
    with cols[i % 3]:
        if st.button(label, key=f"quick_{i}"):
            st.session_state.pending_prompt = full_q

st.markdown("<div style='height:20px; border-bottom:0.5px solid #E8E5DE; margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── DISPLAY CHAT HISTORY ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if available
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View sources", expanded=False):
                for src in message["sources"]:
                    name = os.path.basename(src.get("source", "Unknown"))
                    page = src.get("page", "?")
                    st.markdown(
                        f"<p style='font-size:12px; color:#8a8478; margin:2px 0;'>📑 <b>{name}</b> — Page {page + 1}</p>",
                        unsafe_allow_html=True
                    )

# ── CHAT INPUT + LOGIC ────────────────────────────────────────────────────────
# Resolve prompt: pending (from quick button) takes priority over typed input
pending = st.session_state.pop("pending_prompt", None)
typed = st.chat_input("Ask Lexi a legal question...")
prompt = pending or typed

if prompt:
    # If act filter is set, prepend it as a context hint
    if act_filter != "All Acts":
        prompt_with_filter = f"[Focus on the {act_filter} only] {prompt}"
    else:
        prompt_with_filter = prompt

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Lexi's response
    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ GROQ_API_KEY is missing. Please add it to your .env file.")
        else:
            try:
                with st.spinner("Lexi is reading the law acts..."):
                    vector_store = load_knowledge_base()

                chain = build_chain(vector_store)

                # Build LangChain-compatible history (last 6 turns = 3 exchanges)
                lc_history = []
                history_window = st.session_state.chat_history[-6:]
                for h in history_window:
                    if h["role"] == "human":
                        lc_history.append(HumanMessage(content=h["content"]))
                    else:
                        lc_history.append(AIMessage(content=h["content"]))

                # Stream the response token by token
                response_placeholder = st.empty()
                full_response = ""
                source_docs = []

                result = chain.invoke({
                    "input": prompt_with_filter,
                    "chat_history": lc_history,
                })

                full_response = result["answer"]
                source_docs = result.get("context", [])

                response_placeholder.markdown(full_response)

                # Extract and display sources
                sources = []
                seen = set()
                for doc in source_docs:
                    meta = doc.metadata
                    key = (meta.get("source", ""), meta.get("page", ""))
                    if key not in seen:
                        seen.add(key)
                        sources.append({"source": meta.get("source", ""), "page": meta.get("page", 0)})

                if sources:
                    with st.expander("📄 View sources", expanded=False):
                        for src in sources:
                            name = os.path.basename(src["source"])
                            page = src["page"]
                            st.markdown(
                                f"<p style='font-size:12px; color:#8a8478; margin:2px 0;'>📑 <b>{name}</b> — Page {page + 1}</p>",
                                unsafe_allow_html=True
                            )

                # Save to both message stores
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                st.session_state.chat_history.append({"role": "human", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.caption("Try refreshing the page or check your API key.")