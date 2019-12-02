import sentence_transformers as st
import pandas as pd
import numpy as np
import argparse as ap
import torch
import os
import psutil
import h5py as h5

from tqdm import tqdm
from typing import List
from loguru import logger
import gc

def get_ram_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_usage = py.memory_info()[0]/2.**30  # memory use in GB...I think
    return memory_usage

def init_models(model_name: str, device='cuda:0') -> st.SentenceTransformer:
    return st.SentenceTransformer(model_name_or_path=model_name, device=device)

def load_dataset(path: str, column: str = 'text') -> List[str]:
    dataset = list(pd.read_parquet(path, engine='pyarrow', columns=[column])[column].values)
    return dataset

def save_matrix(matrix: np.ndarray, path: str) -> None:
    with h5py.File(path, 'w') as f:
        dataset = f.create_dataset("dataset", data=matrix, chunks=True, shuffle=True, compression='gzip', compression_opts=9)
    
def process(model: st.SentenceTransformer, texts: List[str], batch_size: int) -> torch.FloatTensor:
    batches = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    gc.collect()
    batches = [v.astype(np.float16, copy=False) for v in tqdm(batches, desc='Casting to fp16')]
    gc.collect()
    batches = np.stack(batches)
    return batches

if __name__ == "__main__":
    parser: ap.ArgumentParser = ap.ArgumentParser(prog='encode.py', description='Encodes text into vectors using sentence transformers')
    parser.add_argument("-i", "--input", nargs='?', help="Input file(in parquet)")
    parser.add_argument("-c", "--column", nargs='?', help="Column name with texts")
    parser.add_argument("-o", "--output", nargs='?', help="Name of output file (result matrix will be stored in gzip-compressed hdf5 format)")
    parser.add_argument("-b", "--batch-size", nargs='?', help='Batch size for inference(depends on amount of memory at your device), default is 128', type=int, default=128)
    parser.add_argument("-d", "--device", nargs='?', help='Name of device to use, possible values are gpu, cpu(and tpu in future)')
    parser.add_argument("-m", "--model-name", nargs='?', help='Name of model to use during encoding, chekc the docs to see available models', default='bert-large-nli-stsb-mean-tokens')
    
    args: ap.Namespace = parser.parse_args()
    logger.info("1. Loading dataset")
    dataset = load_dataset(args.input, args.column)
    logger.info("1. Done")
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'gpu':
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    logger.info("2. Loading Model")
    model = init_models(model_name=args.model_name, device=device)
    logger.info("2. Done")
    
    logger.info("3. Encoding")
    tensor = process(model, dataset, batch_size=args.batch_size)
    del dataset
    del model
    logger.info(f"3. Done, final memory used: {get_ram_usage():.4f}GB")
    
    logger.info(f"4. Saving tensor into {args.output}")
    save_matrix(tensor, args.output)
    logger.info("4. Done")