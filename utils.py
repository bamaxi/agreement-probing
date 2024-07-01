import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_NAME = 'ai-forever/ruGPT-3.5-13B'

def load_model_tokenizer_config(model_name=MODEL_NAME, output_attentions=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='cuda:0',
        load_in_8bit=True,
        offload_folder='./_tmp',
        max_memory={0: f'14GB'},
        cache_dir=".cache/huggingface",
        output_attentions=output_attentions
    )
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return model, tokenizer, config