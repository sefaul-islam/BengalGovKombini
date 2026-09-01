"""
Module C: RAG Chat Engine with Conversational Memory
=====================================================
Initialises the Gemma LLM via LangChain, connects to the ChromaDB vector
store, and exposes a stateful chat function that maintains full session
memory so the bot remembers previous turns.

Usage (programmatic):
    from rag_engine import RAGChatEngine
    engine = RAGChatEngine(api_key="...")
    answer = engine.chat("আমার পাসপোর্ট করতে কী কী লাগবে?")
"""

import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION = "gov_services_bd"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.7-flash"  # Gemini Flash via Google AI Studio
LLM_TEMPERATURE = 0.3
RETRIEVER_K = 3  # top-k chunks to retrieve
MEMORY_WINDOW_K = 5  # remember last 5 conversation turns

SYSTEM_PROMPT = """\
You are a helpful, empathetic assistant for rural Bangladeshi citizens. \
You answer questions about government services like NID (জাতীয় পরিচয়পত্র), \
Passports (পাসপোর্ট), and MyGov services.

Rules you MUST follow:
1. Always answer in simple, colloquial Bengali (বাংলা). Use everyday words \
that a person from a village can understand easily.
2. Avoid complex bureaucratic jargon. If you must use an official term, \
explain it in parentheses immediately.
3. Use the provided context to answer the question. If the answer is NOT \
in the context, politely say: "ভাই/আপু, এই বিষয়ে আমার কাছে এখন তথ্য নেই। \
আপনি নিকটস্থ ইউনিয়ন ডিজিটাল সেন্টারে যোগাযোগ করতে পারেন।"
4. At the end of EVERY response, proactively ask a relevant follow-up \
question to guide the user to the next step of their procedure. For example: \
"আপনি কি জানতে চান এরপর কোথায় যেতে হবে?" or \
"আপনার কি আরও কোনো প্রশ্ন আছে এই বিষয়ে?"
5. Be warm and respectful. Use "আপনি" (polite you).
6. If a user greets you, greet them back warmly and ask how you can help.

Context from knowledge base:
{context}

Conversation so far:
{chat_history}
"""

HUMAN_TEMPLATE = "{question}"


# ---------------------------------------------------------------------------
# RAG Chat Engine
# ---------------------------------------------------------------------------
class RAGChatEngine:
    """Stateful conversational RAG engine backed by ChromaDB + Gemma."""

    def __init__(
        self,
        api_key: str,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        llm_model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        retriever_k: int = RETRIEVER_K,
        memory_window_k: int = MEMORY_WINDOW_K,
    ):
        self.api_key = api_key
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

        # --- Embeddings ---
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        # --- Vector store ---
        logger.info("Connecting to ChromaDB at %s", chroma_dir)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )
        doc_count = self.vector_store._collection.count()
        logger.info("Vector store loaded with %d documents.", doc_count)

        if doc_count == 0:
            logger.warning(
                "⚠ Vector store is empty! Run ingest.py first to populate it."
            )

        # --- Retriever ---
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k},
        )

        # --- LLM ---
        logger.info("Initialising LLM: %s (temp=%.2f)", llm_model, temperature)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature,
        )

        # --- Memory ---
        self.memory = ConversationBufferWindowMemory(
            k=memory_window_k,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # --- Prompt ---
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
            ]
        )

        # --- Conversational Retrieval Chain ---
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False,
        )

        logger.info("✓ RAG Chat Engine initialised successfully.")

    def chat(self, user_message: str) -> dict:
        """
        Send *user_message* through the RAG chain and return a dict:
        {
            "answer": str,
            "source_documents": list[Document],
        }
        """
        logger.info("User: %s", user_message[:120])
        result = self.chain.invoke({"question": user_message})
        answer = result.get("answer", "")
        sources = result.get("source_documents", [])
        logger.info("Bot: %s", answer[:120])
        return {"answer": answer, "source_documents": sources}

    def chat_stream(self, user_message: str):
        """
        Stream response tokens one-by-one. This bypasses the
        ConversationalRetrievalChain (which makes 2 LLM calls) and instead
        does a single retrieval + single LLM call with streaming.

        Usage:
            for token in engine.chat_stream("..."):
                print(token, end="")
            sources = engine.last_source_docs  # available after streaming ends
        """
        logger.info("User (stream): %s", user_message[:120])

        # 1. Retrieve relevant documents
        docs = self.retriever.invoke(user_message)
        self.last_source_docs = docs
        context = "\n\n".join(doc.page_content for doc in docs)

        # 2. Format chat history from memory
        history_messages = self.memory.load_memory_variables({}).get(
            "chat_history", []
        )
        chat_history_str = ""
        for msg in history_messages:
            role = "User" if msg.type == "human" else "Assistant"
            chat_history_str += f"{role}: {msg.content}\n"

        # 3. Build prompt messages
        messages = self.prompt.format_messages(
            context=context,
            chat_history=chat_history_str,
            question=user_message,
        )

        # 4. Stream LLM response (single call)
        full_response = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content or ""
            if token:
                full_response += token
                yield token

        # 5. Save to memory after streaming completes
        self.memory.save_context(
            {"question": user_message},
            {"answer": full_response},
        )
        logger.info("Bot (stream): %s", full_response[:120])

    def reset_memory(self) -> None:
        """Clear conversation memory for a fresh session."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def get_memory_messages(self) -> list:
        """Return the current conversation history from memory."""
        return self.memory.load_memory_variables({}).get("chat_history", [])


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Set GOOGLE_API_KEY environment variable first.")
        return

    engine = RAGChatEngine(api_key=api_key)

    print("\n💬  RAG Chat Engine (type 'quit' to exit, 'reset' to clear memory)\n")
    while True:
        user_input = input("আপনি: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "বাহির"):
            print("ধন্যবাদ! আসসালামু আলাইকুম। 👋")
            break
        if user_input.lower() == "reset":
            engine.reset_memory()
            print("🔄 কথোপকথন রিসেট হয়েছে।\n")
            continue

        result = engine.chat(user_input)
        print(f"\n🤖 সহকারী: {result['answer']}\n")

        if result["source_documents"]:
            print("  📎 তথ্যসূত্র:")
            seen_urls = set()
            for doc in result["source_documents"]:
                url = doc.metadata.get("source_url", "N/A")
                if url not in seen_urls:
                    seen_urls.add(url)
                    print(f"     • {url}")
            print()


if __name__ == "__main__":
    main()
