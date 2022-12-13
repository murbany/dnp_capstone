# import torch
from transformers import pipeline

def init():
    # model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    generator = pipeline('text-generation', model='facebook/opt-125m', device=-1)

    # return model, tokenizer, generator
    return generator
    
