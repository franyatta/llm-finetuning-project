import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import yaml
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    # Get the directory containing the script
    script_dir = Path(__file__).parent.absolute()
    config_file = Path(script_dir) / config_path
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_file}. Please make sure config.yaml is in the same directory as train.py")

def load_model_and_tokenizer(config: Dict[str, Any]):
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    return model, tokenizer

def prepare_dataset(config: Dict[str, Any], tokenizer):
    dataset = load_dataset(config['dataset_name'])
    
    def tokenize_function(examples):
        return tokenizer(
            examples[config['text_column']], 
            truncation=True, 
            padding='max_length', 
            max_length=config['max_length']
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized_dataset

def main():
    print("Starting the training process...")
    
    try:
        config = load_config('config.yaml')
        print("Successfully loaded config.")
        
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(config)
        print(f"Loaded {config['model_name']} model and tokenizer.")
        
        print("Preparing dataset...")
        dataset = prepare_dataset(config, tokenizer)
        print("Dataset preparation complete.")
        
        # Create output directory if it doesn't exist
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config['num_epochs'],
            per_device_train_batch_size=config['batch_size'],
            logging_dir='./logs',
            logging_steps=100,
            save_steps=500,
            learning_rate=config['learning_rate']
        )
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'] if 'validation' in dataset else None
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Training complete. Saving model...")
        final_output_dir = output_dir / 'final_model'
        model.save_pretrained(str(final_output_dir))
        tokenizer.save_pretrained(str(final_output_dir))
        print(f"Model saved to {final_output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()