import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    config = load_config('config.yaml')
    
    model, tokenizer = load_model_and_tokenizer(config)
    
    dataset = prepare_dataset(config, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        learning_rate=config['learning_rate']
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else None
    )
    
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(os.path.join(config['output_dir'], 'final_model'))
    tokenizer.save_pretrained(os.path.join(config['output_dir'], 'final_model'))

if __name__ == '__main__':
    main()