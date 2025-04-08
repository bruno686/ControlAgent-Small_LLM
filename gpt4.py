from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import openai

class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1,
                 engine='gpt-4o-2024-08-06', frequency_penalty=0, presence_penalty=0,
                 stop=None, rstrip=False, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine  
        model_name = "/home/kaiyu/cheers/LLMs/qwen2.5-1.5b-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def complete(self, prompt):
        if self.rstrip:
            prompt = prompt.rstrip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in control engineering design. "
                    "Respond only using a valid object. "
                    "For example: {\"response\": \"your answer here\"}"
                )
            },
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # temperature = max(self.temperature, 1e-5)
        # do_sample = self.temperature > 0.0

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            # temperature=temperature,
            do_sample=False
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # # 打印 response 进行调试
        # print("Response:", response)
        # print("Response Type:", type(response))

        # 删除可能存在的 ```json 标记
        response = response.replace("```json", "").replace("```", "").strip()
        print(response)
        return response