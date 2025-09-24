"""
======================================================================
KDPS ---

Implementation of K-grained Decision Potential Surface

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 16 九月 2025
======================================================================
"""


# ------------------------ Code --------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import os
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

def compute_k_dps(
        prompts,
        K=10, batch_size=5,
        model_name='gpt2',
        max_new_tokens=20,
        sampling_strategy='nucleus', top_p=0.9,
        output_dir='k_dps_outputs',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        ):
    """
    Compute the K-Grained Decision Potential (Phi_f^K(x)) for a list of input prompts and save prompts, generated samples, and log probabilities.
    
    Args:
    - prompts (List[str]): List of input prompts x.
    - K (int): Number of samples to draw from P_f(.|x) per prompt.
    - batch_size (int): Batch size for sequence generation.
    - model_name (str): HuggingFace model name (e.g., 'gpt2').
    - max_new_tokens (int): Length Nr of generated sequences.
    - sampling_strategy (str): Sampling method ('nucleus' or 'topk').
    - top_p (float): Top-p probability for nucleus sampling.
    - output_dir (str): Directory to save output JSON files.
    - device (str): Device to run the model on.
    
    Returns:
    - List[float]: List of Phi_f^K(x) = (log P(y1*^K | x) - log P(y2*^K | x))^2 for each prompt.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    phi_k_values = []
    output_data = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts,desc="PROMPT Infer Stage:")):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        input_len = input_ids.shape[1]
        
        num_batches = (K + batch_size - 1) // batch_size
        all_sequences = []
        all_log_probs = []
        generated_texts = []
        
        for batch_idx in tqdm(range(num_batches), desc=f"Batches for Index: {prompt_idx}"):
            current_batch_size = min(batch_size, K - batch_idx * batch_size)
            input_ids_batch = input_ids.repeat(current_batch_size, 1)
            
            with torch.no_grad():
                gen_args = {
                    'input_ids': input_ids_batch,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': True,
                    'num_return_sequences': current_batch_size,
                    'output_scores': True,
                    'return_dict_in_generate': True,
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id
                }
                
                if sampling_strategy == 'nucleus':
                    gen_args['top_p'] = top_p
                elif sampling_strategy == 'topk':
                    gen_args['top_k'] = 50  # Default top-k value, adjustable
                
                gen_outputs = model.generate(**gen_args)
            
            sequences = gen_outputs.sequences
            all_sequences.append(sequences)
            
            for k in range(current_batch_size):
                seq_logp = 0.0
                for t in range(max_new_tokens):
                    pos = t
                    if pos >= len(gen_outputs.scores):
                        break
                    score = gen_outputs.scores[pos][k]
                    logits = score.unsqueeze(0)
                    log_probs_t = torch.log_softmax(logits, dim=-1)
                    chosen_token_id = sequences[k, input_len + t]
                    seq_logp += log_probs_t[0, chosen_token_id].item()
                all_log_probs.append(seq_logp)
                # Decode generated sequence
                generated_text = tokenizer.decode(sequences[k, input_len:], skip_special_tokens=True)
                generated_texts.append(generated_text)
        
        all_sequences = torch.cat(all_sequences, dim=0)[:K]
        log_probs_tensor = torch.tensor(all_log_probs)[:K]
        top_logps, top_indices = torch.topk(log_probs_tensor, 2)
        delta = top_logps[0] - top_logps[1]
        phi_k = delta ** 2
        phi_k_values.append(phi_k)
        
        # Collect data for this prompt
        prompt_data = {
            'prompt': prompt,
            'phi_k': phi_k.item(),
            'generated_samples': [
                {'text': generated_texts[i], 'log_prob': all_log_probs[i]}
                for i in range(len(generated_texts))
            ]
        }
        output_data.append(prompt_data)
    
    # Save output to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(output_dir, f'k_dps_output_{timestamp}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    return phi_k_values

def main():
    parser = argparse.ArgumentParser(description="Compute K-DPS for a list of prompts and save outputs.")
    parser.add_argument("--prompts", type=str, required=True, help="List of input prompts")
    parser.add_argument("--K", type=int, default=10, help="Number of samples K")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max new tokens Nr")
    parser.add_argument("--sampling", type=str, choices=['nucleus', 'topk'], default="nucleus", help="Sampling strategy")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--output_dir", type=str, default="k_dps_outputs", help="Directory to save output JSON files")
    
    args = parser.parse_args()

    if args.prompts=="wikipedia_mini":
        dataset = load_dataset(
            "philschmid/easyrag-mini-wikipedia",
            "documents",
            split="full",
            )
        textls=dataset["document"]
        newtextls=[]
        for x in textls:
            if len(x)<args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls=newtextls
    else:
        return -1
    
    phi_values = compute_k_dps(
        textls,
        args.K,
        args.batch_size,
        args.model,
        args.max_tokens,
        args.sampling,
        args.top_p,
        args.output_dir
    )

if __name__ == "__main__":
    main()
