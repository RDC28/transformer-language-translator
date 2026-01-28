from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 150, # Reduced seq_len for speed
        "d_model": 128, # Tiny model for CPU
        "d_ff": 512,
        "n_layers": 2,
        "n_heads": 4,
        "dataset_limit": 20000, # Increased to 12000
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "checkpoints", # Cleaned up name
        "model_basename": "tmodel_",
        "preload": None, # Start from scratch
        "tokenizer_file": "tokenizers/tokenizer_{0}.json", # Updated path
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder'] # Direct use, no weird prefixing
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])