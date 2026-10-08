from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')

            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f}s')

            prev_time = time.time()
        
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args:ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            **params
        )

        tokenizer = SentencePieceProcessor() # the tokenizer downloaded is of this
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device.type == 'cuda':
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)
        
        model = Transformer(model_args).to(device)

        if load_model:
            checkpoint.pop('rope.freqs', None) # safer than del
            model.load_state_dict(checkpoint, strict = True)
            print(f'Loaded state dict in {(time.time() - prev_time):.2f}s')
        
        return LLaMA(model, tokenizer, model_args)

        
if __name__ == '__main__':
    torch.manual_seed(42)

    prompt = [
        ""
    ]

    model = LLaMA.build(
        checkpoints_dir = '/project2/vsharan_1861/team_mamba_out/llama-2-7b',
        tokenizer_path = '/project2/vsharan_1861/team_mamba_out/tokenizer.model',
        load_model = True,
        max_seq_len = 1024,
        max_batch_size = 2,
    )

    # Inference the model
