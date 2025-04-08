from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import openai
import outlines
from outlines.models.transformers import Transformers


class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1,
                 engine='gpt-4o-2024-08-06', frequency_penalty=0, presence_penalty=0,
                 stop=None, rstrip=False, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine  
        # self.model = outlines.models.transformers("../LLMs/qwen2.5-1.5b-instruct", device_map="cuda", )

        self.tokenizer = AutoTokenizer.from_pretrained("../LLMs/qwen2.5-1.5b-instruct", trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "../LLMs/qwen2.5-1.5b-instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,   # 推荐 float16 节省显存
            device_map="cuda"            # 自动分配到 GPU
        )

        # Wrap with outlines
        self.model = Transformers(self.base_model, self.tokenizer)

    def complete(self, prompt, schema):
        if self.rstrip:
            prompt = prompt.rstrip()

        system_prompt = "You are an expert in control engineering design."
        user_prompt = prompt

        # Qwen2-Chat uses this format
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        generator = outlines.generate.json(self.model, schema)
        response = generator(full_prompt)
        return response