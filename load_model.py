import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer

def init():
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)

    return model, tokenizer
    
