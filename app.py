"""
Module D: Streamlit Interactive Chat UI
========================================
Provides a continuous chat interface with sidebar API key input,
session-state managed chat history, and source citation display.

Run:
    streamlit run app.py --server.port 8501
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGChatEngine

# Load API key from .env file automatically
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🇧🇩 সরকারি সেবা সহকারী",
    page_icon="🇧🇩",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished, Bengali-friendly UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"], .stMarkdown, .stChatMessage, .stTextInput input {
        font-family: 'Noto Sans Bengali', 'Segoe UI', sans-serif !important;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #006a4e 0%, #004d3a 50%, #00332a 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,106,78,0.3);
    }
    .hero-banner h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .hero-banner p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Source citation */
    .source-citation {
        background: #f0f7f4;
        border-left: 3px solid #006a4e;
        padding: 0.5rem 0.8rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #555;
    }
    .source-citation a {
        color: #006a4e;
        text-decoration: none;
    }
    .source-citation a:hover {
        text-decoration: underline;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faf9 0%, #e8f0ec 100%);
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – API Key & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Flag_of_Bangladesh.svg/320px-Flag_of_Bangladesh.svg.png",
        width=180,
    )
    st.markdown("### ⚙️ সেটিংস")

    # Show API key status (loaded from .env)
    if API_KEY:
        st.success("✅ API Key লোড হয়েছে (.env থেকে)")
    else:
        st.error("❌ API Key পাওয়া যায়নি। .env ফাইলে GOOGLE_API_KEY সেট করুন।")

    st.divider()

    st.markdown("### 📖 ব্যবহার নির্দেশিকা")
    st.markdown(
        """
        1. নিচের চ্যাট বক্সে আপনার **প্রশ্ন** লিখুন
        2. সহকারী আপনাকে **বাংলায়** উত্তর দেবে

        **যেসব বিষয়ে জানতে পারবেন:**
        - 🪪 জাতীয় পরিচয়পত্র (NID)
        - 🛂 পাসপোর্ট আবেদন
        - 🏛️ MyGov সেবাসমূহ
        """
    )

    st.divider()

    if st.button("🔄 নতুন কথোপকথন শুরু করুন", use_container_width=True):
        st.session_state.messages = []
        if "engine" in st.session_state:
            st.session_state.engine.reset_memory()
        st.rerun()

    st.divider()
    st.caption("তৈরি করেছে: 🇧🇩 সরকারি সেবা RAG সিস্টেম")

# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🇧🇩 সরকারি সেবা সহকারী</h1>
        <p>পাসপোর্ট, জাতীয় পরিচয়পত্র ও সরকারি সেবা সম্পর্কে জানুন — সহজ বাংলায়</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    st.session_state.engine = None

# ---------------------------------------------------------------------------
# Engine Initialisation
# ---------------------------------------------------------------------------
def initialise_engine(key: str) -> RAGChatEngine | None:
    """Create (or re-create) the RAG engine with the given API key."""
    try:
        engine = RAGChatEngine(api_key=key)
        return engine
    except Exception as exc:
        st.error(f"ইঞ্জিন শুরু করতে সমস্যা হয়েছে: {exc}")
        return None


# Initialise engine when API key is available from .env
if API_KEY and st.session_state.engine is None:
    with st.spinner("🔄 সিস্টেম প্রস্তুত হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন..."):
        st.session_state.engine = initialise_engine(API_KEY)

# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    # Show welcome message
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            """
            **আসসালামু আলাইকুম!** 👋

            আমি আপনার **সরকারি সেবা সহকারী**। আমি আপনাকে নিচের বিষয়গুলোতে সাহায্য করতে পারি:

            - 🛂 **পাসপোর্ট** — আবেদন, নবায়ন, প্রয়োজনীয় কাগজপত্র
            - 🪪 **জাতীয় পরিচয়পত্র (NID)** — সংশোধন, নতুন করা, হারানো কার্ড
            - 🏛️ **MyGov সেবা** — অনলাইন সরকারি সেবা

            আপনার প্রশ্ন করুন, আমি সহজ বাংলায় উত্তর দেব! 😊
            """
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        # Show sources if available
        if msg["role"] == "assistant" and msg.get("sources"):
            sources_html = "<div class='source-citation'>📎 <strong>তথ্যসূত্র:</strong><br>"
            for src_url in msg["sources"]:
                sources_html += f"• <a href='{src_url}' target='_blank'>{src_url}</a><br>"
            sources_html += "</div>"
            st.markdown(sources_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন... (যেমন: পাসপোর্ট করতে কী লাগে?)"):
    # Validate API key
    if not API_KEY:
        st.warning("⚠️ .env ফাইলে **GOOGLE_API_KEY** সেট করুন।")
        st.stop()

    if st.session_state.engine is None:
        st.error("❌ সিস্টেম প্রস্তুত হয়নি। API Key চেক করুন।")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 উত্তর খুঁজছি..."):
            try:
                result = st.session_state.engine.chat(prompt)
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                # Extract unique source URLs
                source_urls = []
                seen = set()
                for doc in source_docs:
                    url = doc.metadata.get("source_url", "")
                    if url and url not in seen:
                        seen.add(url)
                        source_urls.append(url)

                st.markdown(answer)

                # Display source citations
                if source_urls:
                    sources_html = "<div class='source-citation'>📎 <strong>তথ্যসূত্র:</strong><br>"
                    for src_url in source_urls:
                        sources_html += (
                            f"• <a href='{src_url}' target='_blank'>{src_url}</a><br>"
                        )
                    sources_html += "</div>"
                    st.markdown(sources_html, unsafe_allow_html=True)

                # Save to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_urls,
                    }
                )

            except Exception as exc:
                error_msg = f"দুঃখিত, একটি সমস্যা হয়েছে: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
