import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from accelerate import Accelerator
import logging
import random

# Configure logging
logging.basicConfig(filename='errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class LlamaInference:
    def __init__(self, model_path, cache_dir):
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()
        
        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):
        # Prepare the input using the chat template
        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.9,
                do_sample=True
            )
        
        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()
    
class AyaInference:
    def __init__(self, model_path, cache_dir):
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()
        
        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):
        # Prepare the input using the chat template
        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.75,
                do_sample=True
            )
        
        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()
    

class Qwen2Inference:
    def __init__(self, model_path, cache_dir):
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()
        
        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):
        # Prepare the input using the chat template
        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        
        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                do_sample=True

            )
        
        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()


class MistralInference:
    def __init__(self, model_path, cache_dir):

        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()

        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):

        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True
            )

        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()

class PhiInference:
    def __init__(self, model_path, cache_dir):

        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.ckpt_name = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        )


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()

        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):

        if 'mini' not in self.ckpt_name:
            combined_content = input_text[0]['content'] + " \n" + input_text[1]['content']
            input_text = [
                        {
                            "role": "user",
                            "content": combined_content
                        }
                    ]
        # Prepare the input using the chat template
        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True
            )

        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()

class GemmaInference:
    def __init__(self, model_path, cache_dir):
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=cache_dir
        )

        self.base_model.eval()

        self.base_model, self.tokenizer = self.accelerator.prepare(self.base_model, self.tokenizer)

    def generate_response(self, input_text):
        # Prepare the input using the chat template
        combined_content = input_text[0]['content'] + " \n" + input_text[1]['content']
        input_text = [
                    {
                        "role": "user",
                        "content": combined_content
                    }
                ]
        chat_text = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
        logging.debug("Inputs : %s", chat_text)

        inputs = self.tokenizer(
            chat_text,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        prompt_padded_len = inputs['input_ids'].shape[1]
        print('Running generation')
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True
            )

        gen_tokens = outputs[:, prompt_padded_len:]
        hypothesis = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        return hypothesis.strip()
