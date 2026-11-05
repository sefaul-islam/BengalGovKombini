# 🇧🇩 সরকারি সেবা সহকারী — Gov Service RAG Chatbot

A Retrieval-Augmented Generation (RAG) conversational system designed for Bangladeshi rural citizens. It crawls government service websites (NID, ePassport, MyGov), embeds the content into a vector database, and serves answers through a chat interface in **simplified, colloquial Bengali**.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  crawler.py │────▶│  ingest.py  │────▶│  ChromaDB    │
│  (Scraping) │     │  (Chunking  │     │  (Vectors)   │
│             │     │  & Embedding)│     │              │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                    ┌─────────────┐             │
                    │  app.py     │◀────────────┘
                    │ (Streamlit  │     ┌──────────────┐
                    │  Chat UI)   │────▶│ rag_engine.py│
                    │             │     │ (Gemini LLM +│
                    └─────────────┘     │  Memory)     │
                                        └──────────────┘
```

| Component | Technology |
|-----------|-----------|
| Web Crawler | BeautifulSoup + requests |
| Embeddings | Google `text-embedding-004` |
| Vector DB | ChromaDB (persistent) |
| LLM | Gemini 3.5 Flash (via Google AI Studio) |
| Memory | LangChain ConversationBufferWindowMemory |
| UI | Streamlit |
| Deployment | Docker + Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- A **Google AI Studio API Key** — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Step 1: Clone & Configure

```bash
git clone <your-repo-url>
cd BengalGovKombini

# Set your API key
export GOOGLE_API_KEY="your-api-key-here"
```

### Step 2: Crawl Government Websites

```bash
docker-compose --profile tools run --rm crawler
```

This will:
- Crawl ePassport, NID, and MyGov websites (depth=2)
- Save structured data to `/app/data/scraped_data.json`

### Step 3: Ingest Data into Vector Database

```bash
docker-compose --profile tools run --rm ingest
```

This will:
- Read the scraped JSON
- Chunk text (800 chars, 150 overlap)
- Embed with Google's `text-embedding-004`
- Store vectors in ChromaDB

### Step 4: Launch the Chat Application

```bash
docker-compose up -d
```

Then open your browser at: **http://localhost:8501**

Enter your Google API Key in the sidebar and start chatting!

### Stop the Application

```bash
docker-compose down
```

> **Note:** Your embeddings persist in a Docker volume (`chroma_data`). You only need to re-run the crawler and ingest steps when you want to refresh the knowledge base.

---

## 🖥️ Running Locally (Without Docker)

You can use the provided `run.sh` script to quickly start or manage the application:
```bash
bash run.sh          # Start the Streamlit app
bash run.sh crawl    # Re-crawl websites then ingest & start
bash run.sh ingest   # Re-ingest data then start
```

Alternatively, you can run the steps manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-api-key-here"

# Step 1: Crawl
python crawler.py --output scraped_data.json --depth 2

# Step 2: Ingest
python ingest.py --input scraped_data.json

# Step 3: Run the app
streamlit run app.py --server.port 8501
```

---

## 📁 Project Structure

```
Hackathon/
├── app.py              # Streamlit chat UI
├── crawler.py          # Recursive web crawler
├── ingest.py           # Data chunking & vector ingestion
├── rag_engine.py       # RAG chain with conversational memory
├── requirements.txt    # Python dependencies (pinned)
├── Dockerfile          # Container build instructions
├── docker-compose.yml  # Multi-service orchestration
├── README.md           # This file
├── chroma_db/          # ChromaDB persistent storage (auto-created)
└── scraped_data.json   # Crawled data output (auto-created)
```

---

## 🛠️ Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI Studio API key | Yes |

### Tunable Parameters

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `CHUNK_SIZE` | `ingest.py` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 150 | Overlap between chunks |
| `RETRIEVER_K` | `rag_engine.py` | 5 | Number of chunks retrieved |
| `MEMORY_WINDOW_K` | `rag_engine.py` | 15 | Conversation turns remembered |
| `LLM_MODEL` | `rag_engine.py` | `gemini-3.5-flash` | LLM model name |
| `LLM_TEMPERATURE` | `rag_engine.py` | 0.3 | Response creativity |

---

## 💬 Example Conversation

```
👤 আপনি: পাসপোর্ট করতে কী কী কাগজপত্র লাগে?

🤖 সহকারী: আসসালামু আলাইকুম! পাসপোর্ট করতে আপনার নিচের
   কাগজপত্রগুলো লাগবে:
   ১. জাতীয় পরিচয়পত্র (NID কার্ড)
   ২. জন্ম নিবন্ধন সনদ
   ৩. পাসপোর্ট সাইজের ছবি
   ...
   আপনি কি জানতে চান কোথায় আবেদন করতে হবে?

👤 আপনি: হ্যাঁ, কোথায় আবেদন করব?

🤖 সহকারী: আপনি নিকটস্থ আঞ্চলিক পাসপোর্ট অফিসে আবেদন
   করতে পারেন। অনলাইনেও আবেদন করা যায়...
```

---

## 📝 License

This project is for educational and hackathon purposes. Government website content belongs to the respective Bangladeshi government agencies.
