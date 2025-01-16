# utils.py

import pickle


# Save the model and optimizer state
def save_checkpoint(model, opt_state, filepath="checkpoint.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump((model, opt_state), f)


def load_checkpoint(filepath="checkpoint.pkl"):
    with open(filepath, "rb") as f:
        model, opt_state = pickle.load(f)
    return model, opt_state
