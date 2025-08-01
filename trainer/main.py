import argparse
import torch
import numpy as np
import time
import yaml
import wandb
from pathlib import Path
from tqdm import tqdm
from bpe.tokenizer import Tokenizer
import os

# --- Block 1: Import Your Custom Modules ---
# Import all the components you've built in the previous sections.
# from modules import TransformerLM
# from trainer.data_loading import get_batch
# from optimizer.cosine_schedule import get_lr_cosine_schedule
# from optimizer.adamw import AdamW
# from trainer.checkpointing import save_checkpoint, load_checkpoint
# from modules.loss import cross_entropy
# from optimizer.gradient_clipping import gradient_clipping
from modules.transformer_lm import TransformerLM
from trainer.data_loading import get_batch
from optimizer.cosine_schedule import get_lr_cosine_schedule
from optimizer.adamw import AdamW
from trainer.checkpointing import save_checkpoint, load_checkpoint
from modules.loss import cross_entropy
from optimizer.gradient_clipping import gradient_clipping


# --- Block 2: Helper Functions ---
# These functions handle setup tasks like detecting the device and loading the config.
def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    # Your implementation to load a YAML file goes here.
    with open(path, "r") as f:
        return yaml.safe_load(f)


def encode(tokenizer: Tokenizer, input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        token_ids_generator = tokenizer.encode_iterable(f)
        all_token_ids = list(tqdm(token_ids_generator, desc="Tokenizing"))
    print(f"Saving token IDs to '{output_path}'...")
    # Save the final array to disk with the recommended dtype
    np.save(output_path, np.array(all_token_ids, dtype=np.uint16))
    print("Tokenization complete.")


# --- Block 3: The Main Training Function ---
def main():
    """
    Main function to orchestrate the training of the Transformer Language Model.
    """
    # --- Part A: Configuration and Setup ---
    print("Setting up training...")

    # 1. Load the YAML config file into a dictionary.
    config = load_config("config/base.yaml")
    # 2. Set up your device ('cuda', 'cpu', etc.).

    device = torch.device(detect_device())
    # 1. Tokenizer
    tokenizer: Tokenizer = Tokenizer.from_files(
        vocab_filepath=config["dataset"]["vocab_out"],
        merges_filepath=config["dataset"]["merges_out"],
        special_tokens=config["dataset"]["special_tokens"],
    )
    # 4. Initialize Weights & Biases (wandb) if enabled in your config.
    wandb_flag = config["training"]["wandb"]
    if wandb_flag:
        wandb.init(project=config["training"]["wandb_project"], config=config)
    # --- Part B: Data Loading ---
    # Use np.memmap to load your training and validation datasets efficiently.
    # train_data = ...
    # val_data = ...
    # Load dataset
    print("Loading datasets...")

    if not os.path.exists(config["dataset"]["valid_id_path"]):
        print("Validation tokens not found. Encoding...")
        encode(
            tokenizer=tokenizer,
            input_path=config["dataset"]["valid_path"],
            output_path=config["dataset"]["valid_id_path"],
        )
    else:
        print("Found existing validation tokens.")

    if not os.path.exists(config["dataset"]["train_id_path"]):
        print("Training tokens not found. Encoding...")
        encode(
            tokenizer=tokenizer,
            input_path=config["dataset"]["train_path"],
            output_path=config["dataset"]["train_id_path"],
        )
    else:
        print("Found existing training tokens.")
    train_data = np.memmap(
        config["dataset"]["train_id_path"], dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        config["dataset"]["valid_id_path"], dtype=np.uint16, mode="r"
    )

    # --- Part C: Model and Optimizer Initialization ---
    print("Initializing model and optimizer...")
    # 1. Create an instance of your TransformerLM class using the hyperparameters
    #    from your config file.
    model = TransformerLM(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
        rope_theta=config["model"]["rope_theta"],
        device=device,
    )#.to(device)

    # 2. Create an instance of your AdamW optimizer, passing it the model's parameters.
    optimizer = AdamW(
        params=model.parameters(),
        lr=float(config["optimizer"]["max_learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    # 3. (Optional) Add logic to load a checkpoint if you are resuming training.

    # --- Part D: The Main Training Loop ---
    print("Starting training...")
    # Loop from 0 to the max_iters specified in your config.
    for it in tqdm(range(config["training"]["max_iters"])):

        # 1. Get a batch of data using your get_batch function.
        x, y = get_batch(
            dataset=train_data,
            batch_size=config["training"]["batch_size"],
            context_length=config["model"]["context_length"],
            device=str(device),
        )

        # 2. Calculate the learning rate for this step using your cosine schedule function
        #    and update the learning rate in your optimizer.
        lr = get_lr_cosine_schedule(
            it,
            float(config["optimizer"]["max_learning_rate"]),
            float(config["optimizer"]["min_learning_rate"]),
            config["optimizer"]["warmup_iters"],
            config["optimizer"]["cosine_iters"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 3. Perform a forward pass to get the logits from the model.
        logits = model(x)

        # 4. Calculate the loss using your cross_entropy function.
        #    Remember to reshape the logits and targets correctly.
        loss = cross_entropy(logits, y)

        # 5. Perform the backward pass, clip gradients, and update weights.
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(
            model.parameters(), max_l2_norm=config["optimizer"]["max_l2_norm"]
        )
        optimizer.step()

        # --- Part E: Logging, Validation, and Checkpointing ---

        # Periodically log your training loss and learning rate.
        if wandb_flag:
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": it})
        if it % config["training"]["log_every"] == 0:
            print(f"Step {it}: loss = {loss.item():.4f}, lr = {lr:.6f}")

        # Periodically run a validation step.
        if it > 0 and it % config["training"]["val_every"] == 0:
            # Your validation logic here. Remember to set the model to eval() mode
            # and use torch.no_grad().
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(
                    dataset=val_data,
                    batch_size=config["training"]["batch_size"],
                    context_length=config["model"]["context_length"],
                    device=str(device),
                )
                logits_val = model(x_val)
                loss_val = cross_entropy(logits_val, y_val)
                if wandb_flag:
                    wandb.log({"val/loss": loss_val.item(), "step": it})
                print(f"[Validation] Step {it}: val_loss = {loss_val.item():.4f}")
            model.train()

        # Periodically save a checkpoint.
        if it > 0 and it % config["training"]["checkpoint_every"] == 0:
            # Your logic to call save_checkpoint() goes here.
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=config["training"]["checkpoint_path"],
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
