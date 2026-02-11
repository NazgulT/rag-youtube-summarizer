# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

from backend import config

def get_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL_ID)
        
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

