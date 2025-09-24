from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Optional


def get_text_embeddings(sentences, model_name='sentence-transformers/all-MiniLM-L6-v2',device="cpu"):
    """
    Generates vector representations for a list of text inputs
    
    Args:
        sentences: list, input text list (e.g., ["text1", "text2", ...])
        model_name: str, pre-trained model name (default is all-MiniLM-L6-v2)
    
    Returns:
        torch.Tensor, matrix of sentence embeddings with shape (len(sentences), 384)
    """
    # Define mean pooling function - properly handles attention masks for accurate averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Load tokenizer and model (downloads model on first use, uses cache afterward)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,device_map=device)
    
    # Preprocess text: tokenize, pad, and truncate
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    
    # Generate token embeddings (disable gradient calculation to speed up inference)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling to get sentence embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings (L2 normalization)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

def compute_text_embeddings_llm(
    text_list: List[str],
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    max_length: int = 512,
    pooling_strategy: str = "last",  # "last" (EOS), "first" (BOS), or "mean"
    layer_index: Optional[int] = None,  # None = last layer, or specify layer index
    trust_remote_code: bool = True,
) -> torch.Tensor:
    """
    Compute text embeddings using a causal LM (e.g., LLaMA, GPT).
    
    Args:
        text_list: List of input texts.
        model_name: Pretrained model name or path.
        device: Device to run the model on ("cuda" or "cpu").
        batch_size: Batch size for processing.
        max_length: Maximum sequence length (truncate longer texts).
        pooling_strategy: How to pool token embeddings ("last", "first", or "mean").
        layer_index: Which layer's hidden states to use (None = last layer).
        trust_remote_code: Whether to trust remote code (e.g., for LLaMA).
    
    Returns:
        embeddings: Tensor of shape [num_texts, hidden_size].
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(text_list), batch_size), desc="Computing embeddings"):
        batch_texts = text_list[i : i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states (all layers)
        hidden_states = outputs.hidden_states  # tuple of [batch_size, seq_len, hidden_size]
        
        # Select specific layer (default: last layer)
        if layer_index is not None:
            assert 0 <= layer_index < len(hidden_states), f"Invalid layer index {layer_index}"
            hidden_states = hidden_states[layer_index]
        else:
            hidden_states = hidden_states[-1]  # last layer
        
        # Pooling strategy
        if pooling_strategy == "last":
            # Use [EOS] token embedding (last non-padded token)
            eos_positions = inputs.attention_mask.sum(dim=1) - 1  # [batch_size]
            pooled_embeds = hidden_states[torch.arange(hidden_states.size(0)), eos_positions]
        elif pooling_strategy == "first":
            # Use [BOS] token embedding (first token)
            pooled_embeds = hidden_states[:, 0, :]
        elif pooling_strategy == "mean":
            # Mean pooling (excluding padding tokens)
            mask = inputs.attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            pooled_embeds = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        embeddings.append(pooled_embeds.cpu())  # Move to CPU to save GPU memory
    
    # Concatenate all batches
    embeddings = torch.cat(embeddings, dim=0)  # [num_texts, hidden_size]
    return embeddings

# Test the function
if __name__ == "__main__":
    # Input text list
    texts = [
        "Artificial intelligence is changing the world",
        "Machine learning is a branch of artificial intelligence",
        "Natural language processing enables computers to understand human language"
    ]
    
    # Get vector representations
    embeddings = get_text_embeddings(texts)
    
    # Print result information
    print(f"Number of input texts: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")  # Should output 384 (fixed dimension for all-MiniLM-L6-v2)
    print("Vector representations:")
    print(embeddings)
