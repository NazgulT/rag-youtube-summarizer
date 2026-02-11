import re  #For extracting video id
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting the transcript into chunks
from backend import config


def extract_video_id(url): 
    """Extracts the video ID from a YouTube URL."""
    video_id_match = re.search(r'(?:v=)([^&]+)', url)
    return video_id_match.group(1) if video_id_match else None

def get_transcript(url):
    """Extracts the transcript from a YouTube video."""
    # Extracts the video ID from the URL
    video_id = extract_video_id(url)
 
    # Create a YouTubeTranscriptApi() object
    ytt_api = YouTubeTranscriptApi()
   
    # Fetch the list of available transcripts for the given YouTube video
    transcripts = ytt_api.list(video_id)
   
    transcript = ""
    for t in transcripts:
        # Check if the transcript's language is English
        if t.language_code == 'en':
            if t.is_generated:
                # If no transcript has been set yet, use the auto-generated one
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                # If a manually created transcript is found, use it (overrides auto-generated)
                transcript = t.fetch()
                break  # Prioritize the manually created transcript, exit the loop
   
    return transcript if transcript else None

def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""

    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text to the output string
            #txt += f"Text: {i['text']} Start: {i['start']}\n"
            txt += f" {i.text} \n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
 
    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

