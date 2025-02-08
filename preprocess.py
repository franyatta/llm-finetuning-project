import argparse
from datasets import load_dataset
import re
from typing import List, Dict, Union
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess dataset for language model fine-tuning')
    parser.add_argument('--dataset_name', type=str, required=True, help='HuggingFace dataset name')
    parser.add_argument('--output_file', type=str, default='processed_data.json', help='Output file path')
    parser.add_argument('--min_length', type=int, default=100, help='Minimum text length')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum text length')
    return parser.parse_args()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = text.strip()
    return text

def filter_text(text: str, min_length: int, max_length: int) -> bool:
    if not text:
        return False
    text_length = len(text.split())
    return min_length <= text_length <= max_length

def chunk_long_text(text: str, max_length: int) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word.split()) > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word.split())
        else:
            current_chunk.append(word)
            current_length += len(word.split())
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_dataset(args) -> List[Dict[str, str]]:
    dataset = load_dataset(args.dataset_name)
    processed_data = []
    
    for split in dataset.keys():
        print(f"\nProcessing {split} split...")
        for item in tqdm(dataset[split]):
            if 'text' not in item:
                continue
                
            text = clean_text(item['text'])
            
            if filter_text(text, args.min_length, args.max_length):
                processed_data.append({
                    'text': text,
                    'split': split
                })
            elif len(text.split()) > args.max_length:
                chunks = chunk_long_text(text, args.max_length)
                for chunk in chunks:
                    if filter_text(chunk, args.min_length, args.max_length):
                        processed_data.append({
                            'text': chunk,
                            'split': split
                        })
    
    return processed_data

def save_data(data: List[Dict[str, str]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nProcessed data saved to {output_file}")
    print(f"Total examples: {len(data)}")

def main():
    args = parse_args()
    processed_data = process_dataset(args)
    save_data(processed_data, args.output_file)

if __name__ == "__main__":
    main()