import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

def save_json(name, value):
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    (OUT / name).write_text(json.dumps(value, indent=2, default=convert), encoding='utf-8')
