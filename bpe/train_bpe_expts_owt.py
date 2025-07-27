import time
import pickle
from pathlib import Path
from tests.adapters import run_train_bpe
# uv run python -m bpe.train_bpe_expts_owt

def main():
    """
    Runs the BPE training experiment for the OpenWebText dataset as
    described in Problem (train_bpe_expts_owt).
    """
    print("Starting the OpenWebText BPE training experiment...")

    # --- Configuration ---
    # IMPORTANT: Update this path to point to your actual OpenWebText dataset file.
    input_path = Path("/home/jiayulu/cs336-hw1/data/owt_train.txt") # Changed dataset
    output_dir = Path("/home/jiayulu/cs336-hw1/output")
    output_dir.mkdir(exist_ok=True)

    # Updated vocab size for OpenWebText as per the assignment [cite: 262]
    vocab_size = 32000
    # The special token is likely the same, but confirm with the dataset's README
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
    # Updated output file names for clarity
    vocab_path = output_dir / "owt_vocab.pkl"
    merges_path = output_dir / "owt_merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to: {vocab_path}")

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"Merges saved to: {merges_path}")

    # --- Analyze the Results ---
    longest_token = max(vocab.values(), key=len)
    
    print("\n--- Experiment Results ---")
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Longest token in vocabulary: {longest_token.decode('utf-8', 'ignore')}")
    print(f"Length of the longest token: {len(longest_token)}")

if __name__ == "__main__":
    main()