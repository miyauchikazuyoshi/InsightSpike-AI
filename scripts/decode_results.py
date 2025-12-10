import base64
import sys
import os

def decode_results(input_file="results_base64.txt", output_file="experiments/exp2to4_lite/results/results.zip"):
    print(f"Reading base64 data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please create it and paste the base64 string.")
        return

    with open(input_file, "r") as f:
        b64_data = f.read().strip()
    
    # Remove any potential headers/footers if user copied them by mistake
    if "=" * 80 in b64_data:
        print("Warning: Found separator lines, attempting to clean...")
        parts = b64_data.split("=" * 80)
        # The data should be the longest part
        b64_data = max(parts, key=len).strip()

    print(f"Decoding {len(b64_data)} bytes...")
    try:
        binary_data = base64.b64decode(b64_data)
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(binary_data)
    
    print(f"Success! Saved decoded zip to {output_file}")
    print(f"Size: {len(binary_data)} bytes")
    print("You can now unzip this file to access the results.")

if __name__ == "__main__":
    decode_results()
