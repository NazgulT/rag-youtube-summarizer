# YouTube Summarizer and Q&A Bot

Summarize YouTube videos and ask questions about their content using a RAG pipeline, FAISS and Hugging Face's google/flan-t5-base model.

## Stack

- **Transcripts:** [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- **Orchestration:** [LangChain](https://python.langchain.com/)
- **Embeddings & generation:** [Hugging Face](https://huggingface.co/) (sentence-transformers + transformers)
- **Vector store:** [FAISS](https://ai.meta.com/tools/faiss/)
- **UI:** [Streamlit](https://streamlit.io/)

## Demo

The following session is a summary of the YouTube video on Vector Databases by IBM url = https://www.youtube.com/watch?v=gl1r1XV0SLw

<img width="1426" height="720" alt="YouBot_screenshot" src="https://github.com/user-attachments/assets/b11272bc-2e69-46f6-89c5-1be6f533dcc6" />


## Setup

Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Run

From the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default http://localhost:8501).

## Usage

1. Paste a YouTube video URL and click **Summarize video**.
2. Wait for the transcript to be fetched, summarized, and indexed.
3. Use the **Ask a question** field to query the transcript; answers are generated from the stored chunks and the same Hugging Face model.

## Configuration

See `backend/config.py` and `.env.example`. Key options:

- `EMBEDDING_MODEL` – sentence-transformers model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`).
- `LLM_MODEL_ID` – Hugging Face model for summary and Q&A (default: `google/flan-t5-base`).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` – transcript chunking for RAG.
- `RAG_RETRIEVE_K` – number of chunks retrieved per question.

