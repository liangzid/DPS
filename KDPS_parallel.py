"""
======================================================================
KDPS_PARALLEL ---

Multi-GPU Version of K-DPS Implementation.

    Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
    Copyright © 2025, ANN, all rights reserved.
======================================================================
"""


# ------------------------ Code --------------------------------------
import torch
from vllm import LLM, SamplingParams
import argparse
import json
import os
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import ray
import numpy as np
from transformers import AutoTokenizer

@ray.remote(num_gpus=1)  # Allocate one GPU per task
def compute_k_dps_single_prompt(
    prompt,
    prompt_idx,
    K=20000,
    batch_size=128,
    model_name='gpt2',
    max_new_tokens=20,
    sampling_strategy='nucleus',
    top_p=0.9,
    output_dir='k_dps_outputs'
):
    """
    Compute K-DPS for a single prompt using vLLM for generation.
    
    Args:
    - prompt (str): Input prompt x.
    - prompt_idx (int): Index of the prompt for progress tracking.
    - K (int): Number of samples to draw.
    - batch_size (int): Batch size for generation.
    - model_name (str): Model name (e.g., 'gpt2').
    - max_new_tokens (int): Length of generated sequences.
    - sampling_strategy (str): Sampling method ('nucleus' or 'topk').
    - top_p (float): Top-p for nucleus sampling.
    - output_dir (str): Directory to save output JSON.
    
    Returns:
    - dict: Prompt data including phi_k, generated samples, and any errors.
    """
    # Initialize tokenizer for decoding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Validate prompt
    if not prompt or not prompt.strip():
        print(f"Warning: Skipping empty prompt at index {prompt_idx}")
        return {
            'prompt': prompt,
            'phi_k': 0.0,
            'generated_samples': [],
            'error': 'Empty prompt'
        }
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(input_ids) <= 0:
        print(f"Warning: Skipping prompt '{prompt}' at index {prompt_idx} due to empty tokenized input")
        return {
            'prompt': prompt,
            'phi_k': 0.0,
            'generated_samples': [],
            'error': 'Empty tokenized input'
        }
    
    # Initialize vLLM model
    try:
        llm = LLM(model=model_name, max_model_len=max_new_tokens + len(input_ids), enable_prefix_caching=True)
    except Exception as e:
        print(f"Error initializing model for prompt '{prompt}' at index {prompt_idx}: {str(e)}")
        raise e
        return {
            'prompt': prompt,
            'phi_k': 0.0,
            'generated_samples': [],
            'error': f'Model initialization failed: {str(e)}'
        }
    
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        n=1,  # Generate one sequence per prompt, handle batching manually
        max_tokens=max_new_tokens,
        temperature=1.0 if sampling_strategy in ['nucleus', 'topk'] else 0.0,
        top_p=top_p if sampling_strategy == 'nucleus' else 1.0,
        top_k=50 if sampling_strategy == 'topk' else -1,
        logprobs=1,
    )
    
    num_batches = (K + batch_size - 1) // batch_size
    all_log_probs = []
    generated_texts = []
    
    for batch_idx in tqdm(range(num_batches), desc=f"Batches for Prompt {prompt_idx}"):
        current_batch_size = min(batch_size, K - batch_idx * batch_size)
        prompts_batch = [prompt] * current_batch_size
        
        try:
            with torch.cuda.amp.autocast():
                outputs = llm.generate(prompts_batch, sampling_params)
        except Exception as e:
            print(f"Error generating for prompt '{prompt}' at index {prompt_idx}: {str(e)}")
            raise e
            return {
                'prompt': prompt,
                'phi_k': 0.0,
                'generated_samples': [],
                'error': f'Generation failed: {str(e)}'
            }
        
        for output in outputs:
            generated_text = output.outputs[0].text
            logprobs = output.outputs[0].logprobs
            # print(f"=====================LogProbs: {logprobs}")
            seq_logp = sum(sum(lp.logprob\
                   for lp in logprob.values())\
               for logprob in logprobs) if logprobs else 0.0
            all_log_probs.append(seq_logp)
            generated_texts.append(generated_text)
    
    log_probs_tensor = torch.tensor(all_log_probs)[:K]
    
    if len(log_probs_tensor) < 2:
        print(f"Warning: Insufficient samples for prompt '{prompt}' at index {prompt_idx}")
        return {
            'prompt': prompt,
            'phi_k': 0.0,
            'generated_samples': [{'text': text, 'log_prob': logp} for text, logp in zip(generated_texts, all_log_probs)],
            'error': 'Insufficient samples'
        }
    
    top_logps, _ = torch.topk(log_probs_tensor, 2)
    delta = top_logps[0] - top_logps[1]
    phi_k = delta ** 2
    
    prompt_data = {
        'prompt': prompt,
        'phi_k': phi_k.item(),
        'generated_samples': [
            {'text': generated_texts[i], 'log_prob': all_log_probs[i]}
            for i in range(len(generated_texts))
        ]
    }
    
    return prompt_data

def compute_k_dps(
    prompts,
    K=20000,
    batch_size=128,
    model_name='gpt2',
    max_new_tokens=20,
    sampling_strategy='nucleus',
    top_p=0.9,
    output_dir='k_dps_outputs'
):
    """
    Compute K-DPS for a list of prompts using ray for parallelization.
    
    Args:
    - Same as compute_k_dps_single_prompt, applied to a list of prompts.
    
    Returns:
    - List[float]: List of Phi_f^K(x) values for each prompt.
    """
    # Initialize ray
    if not ray.is_initialized():
        ray.init(num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Submit tasks for each prompt
    futures = [
        compute_k_dps_single_prompt.remote(
            prompt, idx, K, batch_size, model_name, max_new_tokens,
            sampling_strategy, top_p, output_dir
        )
        for idx, prompt in enumerate(prompts)
    ]
    
    # Collect results with progress bar
    output_data = []
    phi_k_values = []
    for future in tqdm(ray.get(futures), desc="Processing Prompts"):
        output_data.append(future)
        phi_k_values.append(future.get('phi_k', 0.0))
    
    # Save output to JSON
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'k_dps_output.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    return phi_k_values

def main():
    parser = argparse.ArgumentParser(description="Compute K-DPS for a list of prompts using vLLM and ray.")
    parser.add_argument("--prompts", type=str, required=True, help="List of input prompts or 'wikipedia_mini' for dataset")
    parser.add_argument("--K", type=int, default=20000, help="Number of samples K")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation per GPU")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=20, help="Max new tokens Nr")
    parser.add_argument("--sampling", type=str, choices=['nucleus', 'topk'], default="nucleus", help="Sampling strategy")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--output_dir", type=str, default="k_dps_outputs", help="Directory to save output JSON files")
    parser.add_argument("--N_dataset", type=int, default=2000, help="Number of Dataset")
    
    args = parser.parse_args()
    
    if args.prompts == "wikipedia_mini":
        dataset = load_dataset(
            "philschmid/easyrag-mini-wikipedia",
            "documents",
            split="full",
        )
        textls = dataset["document"]
        newtextls = []
        for x in textls:
            if len(x) < args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls = newtextls
        # textls=textls
    elif args.prompts == "tulu-3-sft":
        dataset = load_dataset(
            "allenai/tulu-3-sft-mixture",
            split="train",
        )
        datals = dataset["messages"]
        newtextls = []
        for x in datals:
            x=x[0]["content"]
            if len(x) < args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls = newtextls

    elif args.prompts == "openo1-sft":
        dataset = load_dataset(
            "O1-OPEN/OpenO1-SFT",
            split="train",
        )
        newtextls = []
        for x in dataset:
            inst=x["instruction"]
            out=x["output"]
            x=f"User: {inst} Assistant: {out}"
            if len(x) < args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls = newtextls

    elif args.prompts == "hh-rlhf":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            split="train",
        )
        newtextls = []
        for x in dataset:
            inst=x["chosen"]
            x=f"{inst}"
            if len(x) < args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls = newtextls

    elif args.prompts == "alpaca":
        dataset = load_dataset(
            "tatsu-lab/alpaca",
            split="train",
        )
        newtextls = []
        for x in dataset:
            inst=x["text"]
            x=f"{inst}"
            if len(x) < args.max_tokens:
                continue
            elif len(x)>64+args.max_tokens:
                newtextls.append(x[:64])
            else:
                newtextls.append(x[:-args.max_tokens])
        textls = newtextls
    else:
        textls = args.prompts.split(',')

    textls=textls[:args.N_dataset]
    
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
    
    for prompt, phi in zip(textls, phi_values):
        print(f"Prompt: {prompt[:50]}...\nK-DPS value: {phi}\n")
    print(f"Results saved to {os.path.join(args.output_dir, f'k_dps_output_*.json')}")

if __name__ == "__main__":
    main()
