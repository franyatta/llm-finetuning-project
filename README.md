# Language Model Fine-tuning Project

A lightweight framework for fine-tuning small language models, designed to run on basic hardware. Supports multiple datasets for different use cases.

## Project Structure
```
llm-finetuning-project/
├── train.py              # Main training script
├── config.yaml           # AG News dataset configuration
├── config_wikitext.yaml  # WikiText dataset configuration
├── requirements.txt      # Project dependencies
└── results/             # Directory for saved models and checkpoints
```

## Available Datasets

1. **AG News Dataset** (config.yaml)
   - News articles categorized by topic
   - Great for learning news-related content
   - Smaller dataset, faster training

2. **WikiText-2 Dataset** (config_wikitext.yaml)
   - Wikipedia articles with minimal preprocessing
   - Good for general language understanding
   - Medium-sized dataset, balanced training time

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

## Training

Choose your dataset by using the appropriate config file:

1. For AG News dataset:
```bash
python train.py config.yaml
```

2. For WikiText dataset:
```bash
python train.py config_wikitext.yaml
```

The script will:
1. Load and prepare the chosen dataset
2. Initialize the DistilGPT2 model
3. Start the training process with progress bars
4. Save checkpoints regularly
5. Perform evaluation during training
6. Save the final model

To stop training early, use `Ctrl+C`. The model will save its current state before stopping.

## Configuration

Both config files allow you to customize:
- Model parameters (currently using distilgpt2)
- Training parameters (batch size, learning rate, etc.)
- Maximum sequence length
- Number of training epochs

Example configuration for WikiText:
```yaml
# Model configuration
model_name: 'distilgpt2'
dataset_name: 'wikitext'
dataset_config: 'wikitext-2-raw-v1'
text_column: 'text'

# Training configuration
output_dir: './results_wikitext'
num_epochs: 3
batch_size: 8
learning_rate: 0.00002
max_length: 128
```

## Model Outputs

Trained models and checkpoints are saved in the specified output directory:
- `results/checkpoint-{step}/`: Training checkpoints
- `results/final_model/`: Final trained model
- `results_wikitext/`: WikiText model outputs

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