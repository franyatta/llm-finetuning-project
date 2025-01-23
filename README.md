# Language Model Fine-tuning Project

This project provides a framework for fine-tuning small language models using the AG News dataset. It uses PyTorch and the Transformers library to fine-tune DistilGPT2 on news articles for improved text generation.

## Project Structure
```
llm-finetuning-project/
├── train.py          # Main training script
├── config.yaml       # Configuration file
├── requirements.txt  # Project dependencies
└── results/         # Directory for saved models and checkpoints
```

## Features
- Fine-tune DistilGPT2 on the AG News dataset
- Configurable training parameters
- Automatic model checkpointing
- Training progress logging
- Evaluation during training
- Early stopping based on loss

## Setup

1. Prerequisites: Python 3.11 recommended (3.13+ not currently supported by PyTorch)

2. Clone the repository:
```bash
git clone https://github.com/franyatta/llm-finetuning-project.git
cd llm-finetuning-project
```

3. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

4. Install PyTorch first:
```bash
pip3 install torch torchvision torchaudio  # CPU only
# OR for NVIDIA GPU support:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. Install other dependencies:
```bash
pip install 'accelerate>=0.26.0' 'transformers[torch]'
pip install datasets>=2.12.0 pyyaml>=6.0 tqdm>=4.65.0
```

## Configuration

Edit `config.yaml` to customize:
- Model parameters (currently using distilgpt2)
- Training parameters (batch size, learning rate, etc.)
- Maximum sequence length
- Number of training epochs

Example configuration:
```yaml
# Model configuration
model_name: 'distilgpt2'
dataset_name: 'ag_news'
text_column: 'text'

# Training configuration
output_dir: './results'
num_epochs: 3
batch_size: 8
learning_rate: 0.00002
max_length: 128
```

## Training

Start the training process:
```bash
python train.py
```

The script will:
1. Load and prepare the AG News dataset
2. Initialize the DistilGPT2 model
3. Start the training process with progress bars
4. Save checkpoints regularly
5. Perform evaluation during training
6. Save the final model

To stop training early, use `Ctrl+C`. The model will save its current state before stopping.

## Model Outputs

Trained models and checkpoints are saved in the `results` directory:
- `results/checkpoint-{step}/`: Training checkpoints
- `results/final_model/`: Final trained model

## Requirements

- Python 3.11 (recommended)
- PyTorch
- Transformers library
- HuggingFace datasets
- PyYAML
- tqdm
- accelerate

## License

MIT License