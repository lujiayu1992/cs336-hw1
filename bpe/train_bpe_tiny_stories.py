import time
import pickle
from pathlib import Path
from tests.adapters import run_train_bpe

# uv run python -m bpe.train_bpe_tiny_stories
# Training finished in 418.10 seconds.
# Vocabulary saved to: /home/jiayulu/cs336-hw1/output/tinystories_vocab.pkl
# Merges saved to: /home/jiayulu/cs336-hw1/output/tinystories_merges.pkl

# --- Experiment Results ---
# Total vocabulary size: 10000
# Longest token in vocabulary: b' accomplishment'
# Length of the longest token: 15
def main():
    """
    Runs the BPE training experiment for the TinyStories dataset as
    described in Problem (train_bpe_tinystories).
    """
    print("Starting the TinyStories BPE training experiment...")

    # --- Configuration ---
    # IMPORTANT: Update this path to point to your actual TinyStories dataset file.
    # The assignment mentions it might be in a /data directory.
    input_path = Path("/home/jiayulu/cs336-hw1/data/TinyStoriesV2-GPT4-train.txt")
    output_dir = Path("/home/jiayulu/cs336-hw1/output")
    output_dir.mkdir(exist_ok=True) # Create the output directory if it doesn't exist

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # --- Run Training and Measure Time ---
    start_time = time.time()
    
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTraining finished in {duration:.2f} seconds.")

    # --- Serialize the Results ---
    # The assignment asks you to serialize the vocab and merges to disk[cite: 251].
    vocab_path = output_dir / "tinystories_vocab.pkl"
    merges_path = output_dir / "tinystories_merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to: {vocab_path}")

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"Merges saved to: {merges_path}")

    # --- Analyze the Results ---
    # This part helps you answer the questions from the PDF.
    longest_token = max(vocab.values(), key=len)
    
    print("\n--- Experiment Results ---")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Longest token in vocabulary: {longest_token}")
    print(f"Length of the longest token: {len(longest_token)}")

if __name__ == "__main__":
    main()