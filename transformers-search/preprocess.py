import pandas as pd
import numpy as np
import argparse as ap
from loguru import logger

def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("nan", np.nan)
    df = df.dropna()
    return df

def clean_strings(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].str.split()
    df[column] = df[column].astype('str')
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def main(args):
    
    logger.info(f"1. Reading file: {args.input}")
    df: pd.DataFrame = pd.read_csv(args.input, encoding='utf-8')
    original_len = len(df)
    logger.info(f"1. Done!")
    logger.info(f"2. Dropping all the columns except the one specified: {args.column}")
    df = df.loc[:, [args.column]]
    df = clean_strings(df, args.column)
    logger.info(f"2. Done!")
    logger.info(f"3. Removing empty rows and duplicates")
    df = remove_nans(df)
    df = remove_duplicates(df)
    logger.info(f"3. {len(df)} rows left ({(1 - len(df)/original_len) * 100:.4f}% reduction)")
    logger.info(f"3. Done!")
    logger.info(f"4. Writing result into parquet file with {args.compression} compression: {args.output}")
    df.to_parquet(args.output, engine='pyarrow', compression=args.compression, index=False)
    logger.info(f"4. Done!")
    
if __name__ == "__main__":
    parser: ap.ArgumentParser = ap.ArgumentParser(prog='preprocess.py', description="File preprocessing utility, removes empty rows and converts file into parquet format")
    parser.add_argument("-i", "--input", nargs='?', help="Input file with texts, in CSV format")
    parser.add_argument("-c", "--column", nargs='?', help="Column with texts")
    parser.add_argument("-o", "--output", nargs='?', help="Filename where output will be written, in Apache Parquet format")
    parser.add_argument("-z", "--compression", nargs="?", default='snappy', help='Compression algorithm for output, possible values are snappy, gzip, brotli')
    
    args: ap.Namespace = parser.parse_args()
    main(args)
    