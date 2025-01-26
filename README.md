# Language Model Fine-tuning Project

A lightweight framework for fine-tuning small language models, designed to run on basic hardware. Features optimized configurations for both news articles and general text.

## Project Structure
```
llm-finetuning-project/
├── train.py              # Main training script
├── config.yaml           # Optimized AG News configuration
├── config_wikitext.yaml  # Optimized WikiText configuration
├── requirements.txt      # Project dependencies
└── results/             # Directory for saved models and checkpoints
```

## Available Datasets

1. **AG News Dataset** (config.yaml)
   - News articles categorized by topic
   - Optimized for quick training on basic hardware
   - Configuration tuned for shorter text sequences
   - Training time: ~1-2 hours on CPU
   ```yaml
   num_epochs: 2        # Shorter training cycle
   batch_size: 16       # Optimized for shorter texts
   learning_rate: 5e-5  # Faster learning for structured data
   max_length: 64       # Tuned for news article length
   ```

2. **WikiText-2 Dataset** (config_wikitext.yaml)
   - Wikipedia articles with minimal preprocessing
   - Configured for comprehensive language learning
   - Handles longer text sequences effectively
   - Training time: ~3-4 hours on CPU
   ```yaml
   num_epochs: 3        # Fuller training cycle
   batch_size: 4        # Handles longer sequences
   learning_rate: 1e-5  # Stable learning for diverse content
   max_length: 256      # Captures more context
   ```

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

Choose your dataset based on your needs:

1. For news article generation (faster training):
```bash
python train.py config.yaml
```

2. For general text generation (more comprehensive):
```bash
python train.py config_wikitext.yaml
```

Training features:
- Progress bars with loss metrics
- Regular checkpoints
- Automatic evaluation
- Early stopping capability
- Memory-efficient processing

To stop training early, use `Ctrl+C` (the model will save its current state).

## Hardware Requirements

The project is optimized for basic hardware:
- Minimum 8GB RAM
- CPU training supported
- GPU optional (will be used if available)
- ~2GB disk space for models and checkpoints

## Model Outputs

The trained models are saved in separate directories:
- AG News: `results/final_model/`
- WikiText: `results_wikitext/final_model/`

Each directory contains:
- Model weights
- Tokenizer files
- Training checkpoints

## Requirements

- Python 3.11 (recommended)
- PyTorch
- Transformers library
- HuggingFace datasets
- PyYAML
- tqdm
- accelerate

## License

MIT License - Free to use, modify, and distribute with attribution.