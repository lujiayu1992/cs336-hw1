import argparse
import torch
import yaml
from pathlib import Path
from rich.console import Console

# --- Block 1: Import Your Custom Modules ---
# These imports align with your training script's structure.
from modules.transformer_lm import TransformerLM
from bpe.tokenizer import Tokenizer
from trainer.checkpointing import save_checkpoint, load_checkpoint
from einops import reduce
import torch.nn.functional as F
from tqdm import tqdm


# --- Block 2: Helper Functions ---
def detect_device() -> str:
    """Detects the most appropriate compute device available."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --- Block 3: Core Generation Function ---
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    context_length: int,
) -> str:
    """
    Generates text autoregressively using a trained model.

    Args:
        model: The trained TransformerLM model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompt: The initial text to start generation from.

    Returns:
        The generated text, including the original prompt.
    """
    # 1. Set the model to evaluation mode (`model.eval()`) and move it to the correct device.
    model.eval()
    device = detect_device()
    eos_token_id = tokenizer.inverse_vocab.get(b"<|endoftext|>")
    # 2. Encode the input `prompt` into a tensor of token IDs.
    input_ids = tokenizer.encode(prompt)
    # 3. Create the initial `context` tensor, add a batch dimension, and move it to the device.
    context = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    # 4. Use a `torch.no_grad()` block to disable gradient calculations.
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc=" generating next token"):
            context_for_model = (
                context
                if context.size(1) <= context_length
                else context[:, -context_length:]
            )
            logits = model(context_for_model)
            last_token_logits = logits[:, -1, :]
            logits_scaled = last_token_logits / temperature
            top_k_logits, _ = torch.topk(logits_scaled, top_k)
            min_logit_val = top_k_logits[:, -1].unsqueeze(-1)

            # 2. Create a boolean mask for all logits smaller than the k-th logit.
            remove_indices = logits_scaled < min_logit_val

            # 3. Set these logits to negative infinity.
            logits_scaled.masked_fill_(remove_indices, -torch.inf)
            probs = F.softmax(logits_scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).to(device)
            context = torch.cat((context, next_token), dim=-1)
            if next_token.item() == eos_token_id:
                break
    return tokenizer.decode(context.squeeze(0).tolist())


def main():
    """
    Main function to parse arguments, load resources, and run generation.
    """
    # 1. Tokenizer
    config = load_config("config/base.yaml")
    tokenizer: Tokenizer = Tokenizer.from_files(
        vocab_filepath=config["dataset"]["vocab_out"],
        merges_filepath=config["dataset"]["merges_out"],
        special_tokens=config["dataset"]["special_tokens"],
    )
    device = detect_device()
    model = TransformerLM(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
        rope_theta=config["model"]["rope_theta"],
        device=torch.device(device),
    )
    checkpoint = torch.load(config["generation"]["checkpoint_path"])
    model.load_state_dict(checkpoint["model"])
    console = Console()
    console.print("\n[bold yellow]Generated Sample Text:[/bold yellow]")
    decoded_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=config["generation"]["prompt"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_k=config["generation"]["top_k"],
        context_length=config["model"]["context_length"],
    )
    if decoded_text:
        console.print(decoded_text)


# --- Block 5: Script Entry Point ---
if __name__ == "__main__":
    main()
