from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

def get_llm():
    # We use a relatively small and fast model for demonstration so it runs locally on CPU/GPU without keys
    # google/flan-t5-small or base are excellent instruction tuning models that don't require heavy resources
    model_id = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
