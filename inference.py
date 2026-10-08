from typing import Optional, List
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

        model.eval() # modules default to training mode, which would skip the KV cache

        return LLaMA(model, tokenizer, model_args)

    @torch.inference_mode()
    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        prompt_tokens = [self.tokenizer.encode(prompt, out_type = int, add_bos = True, add_eos = False) for prompt in prompts]

        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        # store prompt and generated tokens
        tokens = torch.full((batch_size, total_len), pad_id, dtype = torch.long, device = device)

        for k, t in enumerate(prompt_tokens):
            # fill initial tokens with the prompt
            tokens[k, :len(t)] = torch.tensor(t, dtype = torch.long, device = device)
        
        eos_reached = torch.zeros(batch_size, dtype = torch.bool, device = device)
        prompt_tokens_mask = tokens != pad_id

        for cur_pos in tqdm(range(1, total_len), desc = 'Generating tokens'):
            # the input token sits at position cur_pos - 1, which is also its KV cache slot and RoPE position
            logits = self.model.forward(tokens[:, cur_pos - 1: cur_pos], cur_pos - 1)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim = -1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim = -1)
            
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # if found eos token on a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            # if all prompts end, break
            if eos_reached.all():
                break
        
        out_tokens = []
        out_text = []

        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):

            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return out_tokens, out_text
    
    def _sample_top_p(self, probs, p):
        # probs: (batch, vocab_size), each row sums to 1
        probs_sort, probs_idx = torch.sort(probs, dim = -1, descending = True)
        probs_sum = torch.cumsum(probs_sort, dim = -1)
        # drop a token if the tokens ranked above it already cover p, so the top token is always kept
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim = -1, keepdim = True)) # renormalize the nucleus
        next_token = torch.multinomial(probs_sort, num_samples = 1) # index into the sorted order
        return torch.gather(probs_idx, -1, next_token) # map back to vocab ids, (batch, 1)

if __name__ == '__main__':
    torch.manual_seed(42)

    prompts = [
        "Chat GPT stands for",
        "Here is one knock knock joke related to a pirate:",
    ]

    model = LLaMA.build(
        checkpoints_dir = '/project2/vsharan_1861/team_mamba_out/llama-2-7b',
        tokenizer_path = '/project2/vsharan_1861/team_mamba_out/tokenizer.model',
        load_model = True,
        max_seq_len = 1024,
        max_batch_size = 2,
    )

    # Inference the model
    out_tokens, out_text = model.text_completion(prompts, max_gen_len = 64)
    for text in out_text:
        print(text)
        print('-' * 50)
