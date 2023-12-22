import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from contextlib import nullcontext
import tiktoken
import numpy as np

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
seed = 10
seed_everything(seed)


device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = '/home/thuannd/Projects/triton-gpt2/model_repository/gpt2/1/student_distill.pt'
ckpt = torch.load(ckpt_path)
model = GPT2LMHeadModel(ckpt['model_args'])
model.load_state_dict(ckpt['model'])
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

start = "Would you proceed especially against Caius Marcius?"
start_ids = tokenizer.encode(text=start)



input_ids = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        y = model.generate(inputs = input_ids, 
                            num_beams=4,
                            do_sample=True,
                            max_new_tokens=50,
                            pad_token_id=50256)
        
        print(tokenizer.decode(y[0].tolist()))

