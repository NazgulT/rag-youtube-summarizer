"""RAG pipeline for Q&A over video transcript using ChromaDB retriever and Hugging Face LLM."""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend import config
from backend.llm import get_llm
from backend.embeddings import create_faiss_index, perform_similarity_search, _get_embeddings
from backend.transcripts import get_transcript, process, chunk_transcript


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


RAG_PROMPT = PromptTemplate.from_template(
    "Use the following transcript excerpts to answer the question. "
    "If the context does not contain relevant information, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)



def answer_question(processed_transcript, user_question):
    """
    Title: Answer User's Question
 
    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.
 
    Args:
        video_id (int): The ID of the YouTube video from which the transcript was fetched.
        user_question (str): The question posed by the user regarding the video.
 
    Returns:
        str: The answer to the user's question or a message indicating that the transcript
             has not been fetched.
    """
    #global fetched_transcript, processed_transcript
 
    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."
 
    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

 
        # Step 2: Initialize LLM for Q&A
        llm = get_llm()
 
        # Step 3: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = _get_embeddings()
        faiss_index = create_faiss_index(chunks, embedding_model)
 
        # Step 4: Set up the Q&A prompt and chain
        #qa_prompt = create_qa_prompt_template()
        #qa_chain = create_qa_chain(llm, qa_prompt)

        qa_chain = (
            RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        relevant_context = perform_similarity_search(faiss_index, user_question)
 
        # Step 6: Generate the answer using FAISS index
        # Generate answer using the QA chain
        #answer = qa_chain.invoke(context=relevant_context, question=user_question)
        answer = qa_chain.invoke({"context": relevant_context, "question": user_question})
        return answer
    
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."