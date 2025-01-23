import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import yaml
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    script_dir = Path(__file__).parent.absolute()
    config_file = Path(script_dir) / config_path
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            # Ensure numeric values are of the correct type
            config['num_epochs'] = int(config['num_epochs'])
            config['batch_size'] = int(config['batch_size'])
            config['learning_rate'] = float(config['learning_rate'])
            config['max_length'] = int(config['max_length'])
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_file}. Please make sure config.yaml is in the same directory as train.py")

def load_model_and_tokenizer(config: Dict[str, Any]):
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Set up padding token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def prepare_dataset(config: Dict[str, Any], tokenizer):
    print("Loading dataset...")
    dataset = load_dataset(config['dataset_name'])
    print("Dataset loaded successfully.")
    
    def tokenize_function(examples):
        # Combine title and description for ag_news dataset
        texts = [f"{title} {desc}" for title, desc in zip(examples['text'], examples['description'])]
        
        result = tokenizer(
            texts,
            truncation=True,
            max_length=config['max_length'],
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels for language modeling (shift input_ids)
        result["labels"] = result["input_ids"].clone()
        
        return result
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing the dataset"
    )
    print("Dataset tokenization complete.")
    
    return tokenized_dataset

def main():
    print("Starting the training process...")
    
    try:
        config = load_config('config.yaml')
        print("Successfully loaded config:", config)
        
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
            per_device_eval_batch_size=config['batch_size'],
            logging_dir='./logs',
            logging_steps=100,
            save_steps=500,
            learning_rate=float(config['learning_rate']),
            evaluation_strategy="steps",
            eval_steps=500,
            warmup_steps=500,
            weight_decay=0.01,
            logging_first_step=True,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            remove_unused_columns=False,  # Important for language modeling
        )
        print("Training arguments:", training_args)
        
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling, not masked
        )
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator
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