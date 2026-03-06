import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline

from app.config import *



def load_llm():

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,use_fast=True,
        trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,#load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=220,
        temperature=0.2,
        repetition_penalty=1.1,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm