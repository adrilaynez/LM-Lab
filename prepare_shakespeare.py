import urllib.request
import re
import os

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "c:/Projects/LM-Lab/data"
OUTPUT_PATH = os.path.join(DATA_DIR, "tinyshakespeare_clean.txt")

def process():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Downloading Tinyshakespeare from {URL}...")
    response = urllib.request.urlopen(URL)
    raw_text = response.read().decode('utf-8')
    
    print(f"Original length: {len(raw_text)} characters")
    
    # Lowercase
    text = raw_text.lower()
    
    # Replace anything that is not a-z, space, or dot with a space
    text = re.sub(r'[^a-z \.]', ' ', text)
    
    # Collapse multiple spaces into a single space
    text = re.sub(r' +', ' ', text)
    
    print(f"Cleaned length: {len(text)} characters")
    
    # Verify exact set of characters
    unique_chars = sorted(list(set(text)))
    print(f"Unique characters ({len(unique_chars)}): {repr(''.join(unique_chars))}")
    
    if len(unique_chars) > 28:
        print("WARNING: Found more than 28 characters!")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
        
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process()
