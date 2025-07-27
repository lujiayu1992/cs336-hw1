import pickle
import argparse
from pathlib import Path

# 256: b'<|endoftext|>'

def load_and_inspect_pickle(file_path, file_description):
    """
    Opens a .pkl file and prints some information about its contents.
    """
    print(f"\n--- Inspecting {file_description} ---")
    try:
        # Open the file in binary read mode ('rb')
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
            print(f"Successfully loaded data from: {file_path}")
            print(f"Type of the loaded data: {type(data)}")

            # --- Inspect the data based on its type ---
            if isinstance(data, dict):
                print(f"Number of items in dictionary: {len(data)}")
                print("First 5 items:")
                for i, (key, value) in enumerate(data.items()):
                    if i >= 5:
                        break
                    print(f"  {key}: {value}")
            elif isinstance(data, list):
                print(f"Number of items in list: {len(data)}")
                print("First 5 items:")
                for i, item in enumerate(data):
                    if i >= 5:
                        break
                    print(f"  {item}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Load and inspect BPE vocab and merges .pkl files.")
    
    # 2. Add arguments for both file paths
    parser.add_argument("vocab_path", type=str, help="The full path to the vocabulary .pkl file.")
    parser.add_argument("merges_path", type=str, help="The full path to the merges .pkl file.")
    
    # 3. Parse the arguments
    args = parser.parse_args()
    
    # 4. Call the inspection function for each file
    load_and_inspect_pickle(args.vocab_path, "Vocabulary File")
    load_and_inspect_pickle(args.merges_path, "Merges File")