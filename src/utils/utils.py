# utils.py

import pickle
import os


# Save the model and optimizer state
def save_checkpoint(model, opt_state, filepath="checkpoint.pkl"):
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump((model, opt_state), f)


def load_checkpoint(filepath="checkpoint.pkl"):
    with open(filepath, "rb") as f:
        model, opt_state = pickle.load(f)
    return model, opt_state
