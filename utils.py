import numpy as np
import random
import torch

from torch.backends import cudnn

def set_global_seed(seed):

    cudnn.deterministic = True
    cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, preprocessor, path):
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'preprocessor': preprocessor
    }
    torch.save(checkpoint, path)

def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    preprocessor = checkpoint['preprocessor']
    return model, preprocessor
