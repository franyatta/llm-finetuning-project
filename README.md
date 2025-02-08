# Language Model Fine-tuning Project

This project provides a framework for fine-tuning small language models using the HuggingFace Transformers library. It includes data preprocessing, model training, and evaluation capabilities.

## Features

- Data preprocessing with text cleaning and chunking
- Configurable model training parameters
- Interactive model evaluation
- Support for HuggingFace datasets
- Perplexity-based model evaluation
- Flexible configuration system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/franyatta/llm-finetuning-project.git
cd llm-finetuning-project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Process your dataset before training:

```bash
python preprocess.py --dataset_name "your_dataset" \
                    --output_file processed_data.json \
                    --min_length 100 \
                    --max_length 1024
```

### Training

Train the model using the processed dataset:

```bash
python train.py --model_name "EleutherAI/pythia-70m" \
                --dataset_name "your_dataset" \
                --num_epochs 3 \
                --batch_size 8
```

### Evaluation

Evaluate the fine-tuned model:

```bash
python evaluate.py --model_path "./output" \
                  --input_file "test_data.txt"
```

## Configuration

The project uses a configuration file (`config.json`) to manage various parameters:

- Model configuration (architecture, size, etc.)
- Training parameters (learning rate, batch size, etc.)
- Data processing settings
- Generation parameters

You can modify these settings in the `config.json` file.

## Project Structure

```
llm-finetuning-project/
├── train.py           # Main training script
├── evaluate.py        # Model evaluation script
├── preprocess.py      # Data preprocessing utilities
├── config.json        # Configuration file
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- tqdm
- numpy

See `requirements.txt` for complete list of dependencies.

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License - feel free to use this project for your own work.
