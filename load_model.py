import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelLoader:
    def __init__(self, base_model_id="mistralai/Mixtral-8x22B-Instruct-v0.1"):

        self.model_id = base_model_id

        # 4bit
        self.bnb_config_4_bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 8bit
        self.bnb_config_8_bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  
            llm_int8_skip_modules=None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.bnb_config_4_bit,
            device_map="auto",
        )
    
    def invoke(self, input: str) -> str:
        self.llm.eval()
        with torch.no_grad():
            model_input = self.tokenizer(input, return_tensors="pt").to("cuda")
            generated_tokens = self.llm.generate(
                **model_input, 
                max_new_tokens=512, 
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id
            )
            # Slice off the input tokens
            new_tokens = generated_tokens[0, model_input['input_ids'].shape[1]:]
            # Decode only the new tokens
            response = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )

            return response
