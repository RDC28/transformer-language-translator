from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 20, # Increased epochs for better convergence
        "lr": 10**-4,
        "seq_len": 350, # Standard sequence length
        "d_model": 512, # Standard Transformer size
        "d_ff": 2048,   # Standard feedforward size
        "n_layers": 6,  # Standard depth
        "n_heads": 8,   # Standard heads
        "dataset_limit": None, # Use full dataset (or set to ~30000 if too slow)
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "checkpoints", # Cleaned up name
        "model_basename": "tmodel_",
        "preload": "latest", # Resume from latest
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