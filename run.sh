#!/bin/bash
# ============================================================================
# 🇧🇩 Gov Service RAG Chatbot — One-click runner
# ============================================================================
# Usage:
#   ./run.sh          — Start the Streamlit app
#   ./run.sh crawl    — Re-crawl websites then ingest & start
#   ./run.sh ingest   — Re-ingest data then start
# ============================================================================

set -e

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ API key loaded from .env"
else
    echo "❌ .env file not found. Create one with: GOOGLE_API_KEY=your-key"
    exit 1
fi

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

# Handle commands
case "${1:-run}" in
    crawl)
        echo "🕷️  Crawling websites (depth 0)..."
        python crawler.py --depth 0 --output scraped_data.json
        echo ""
        echo "📦 Ingesting into ChromaDB..."
        rm -rf chroma_db
        python ingest.py --input scraped_data.json --chroma-dir ./chroma_db --collection gov_services_bd
        echo ""
        echo "🚀 Launching Streamlit..."
        streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
        ;;
    ingest)
        echo "📦 Ingesting into ChromaDB..."
        rm -rf chroma_db
        python ingest.py --input scraped_data.json --chroma-dir ./chroma_db --collection gov_services_bd
        echo ""
        echo "🚀 Launching Streamlit..."
        streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
        ;;
    run|start)
        echo "🚀 Launching Streamlit..."
        streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
        ;;
    *)
        echo "Usage: ./run.sh [crawl|ingest|run]"
        exit 1
        ;;
esac
