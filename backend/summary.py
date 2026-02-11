"""Video summary generation using Hugging Face text-generation LLM."""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.llm import get_llm


SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following video transcript in a paragraph using 5-10 sentences. "
    "Provide an informative summary that captures the main points of the video content.\n\n Transcript:\n{transcript}"
)


def summarize(transcript_text: str) -> str:
    """
    Generate a summary of the given transcript using the configured Hugging Face LLM.
    """
    if not transcript_text or not transcript_text.strip():
        return "No transcript content to summarize."
    # Truncate if very long to stay within model context
    max_chars = 8000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n\n[... transcript truncated ...]"
    chain = SUMMARY_PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"transcript": transcript_text}).strip()
