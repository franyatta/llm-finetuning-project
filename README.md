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

3. Install dependencies:
```bash
pip install -r requirements.txt
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