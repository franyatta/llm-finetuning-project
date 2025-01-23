# Language Model Fine-tuning Project

This project provides a framework for fine-tuning small language models using datasets from HuggingFace.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/franyatta/llm-finetuning-project.git
cd llm-finetuning-project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install PyTorch:
Visit https://pytorch.org/get-started/locally/ and run the appropriate command for your system. For example:
- For CUDA (if you have an NVIDIA GPU):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- For CPU only:
```bash
pip3 install torch torchvision torchaudio
```

4. Install other dependencies:
```bash
pip install transformers>=4.30.0 datasets>=2.12.0 pyyaml>=6.0 tqdm>=4.65.0
```

## Configuration

Modify `config.yaml` to set your desired:
- Model (default: distilgpt2)
- Dataset
- Training parameters

## Training

Run the training script:
```bash
python train.py
```

The fine-tuned model will be saved in the specified output directory.

## Customization

You can customize the training by:
1. Modifying the model architecture in `config.yaml`
2. Adjusting training parameters
3. Using different datasets from HuggingFace

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- HuggingFace datasets

## License

MIT License