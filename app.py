"""Streamlit app: YouTube URL input, summary, and Q&A over transcript."""
import streamlit as st

from backend import config
from backend import transcripts
from backend.rag import answer_question
from backend.summary import summarize

st.set_page_config(page_title="YouBot: YouTube Video Summarizer and Q&A", layout="wide")
st.title("YouBot: YouTube Video Summarizer and Q&A")

# Session state for current video and summary
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "processed_transcript" not in st.session_state:
    st.session_state.processed_transcript = None

# URL input and summarize
url = st.text_input("YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Summarize video"):
    if not url or not url.strip():
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = transcripts.extract_video_id(url)
        if not video_id:
            st.error("Could not parse video ID from URL.")
        else:
            with st.spinner("Fetching transcript..."):
                raw = transcripts.get_transcript(url)
            if raw is None:
                st.error("No English transcript available for this video.")
            else:
                with st.spinner("Processing and summarizing..."):
                    processed = transcripts.process(raw)
                    chunks = transcripts.chunk_transcript(
                        processed,
                        chunk_size=config.CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP,
                    )
                    summary_text = summarize(processed)
                st.session_state.video_id = video_id
                st.session_state.summary = summary_text
                st.session_state.processed_transcript = processed
                st.success("Summary and index ready.")

# Show summary if we have one
if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

# Q&A section (only when we have a video indexed)
if st.session_state.processed_transcript:
    st.subheader("Ask a question about this video")
    question = st.text_input("Your question", key="qa_question")
    if question and question.strip():
        with st.spinner("Searching transcript and generating answer..."):
            answer = answer_question(st.session_state.processed_transcript, question.strip())
        st.write("**Answer:**")
        st.write(answer)
else:
    st.info("Enter a YouTube URL and click **Summarize video** to enable Q&A.")
