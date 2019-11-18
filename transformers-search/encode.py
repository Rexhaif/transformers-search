import sentence_transformers as st
import pandas as pd
import numpy as np
import torch

from typing import List
from loguru import logger
import gc

def init_models(model_name: str, device='cuda:0') -> st.SentenceTransformer:
    return st.SentenceTransformer(model_name_or_path=model, device='cuda:0')

def load_dataset(path: str, column: str = 'text') -> List[str]:
    dataset = list(pd.read_parquet(path, engine='pyarrow', columns=[column])[column].values)
    return dataset
    
def process(model: st.SentenceTransformer, texts: List[str], batch_size: int) -> torch.FloatTensor:
    batches = model.encode(texts, )