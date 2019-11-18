import pandas as pd
import numpy as np
import argparse as ap
from loguru import logger

def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("nan", np.nan)
    df = df.dropna()
    return df

def main(args):
    
    
    logger.info(f"1. Reading file: {args.input}")
    df: pd.DataFrame = pd.read_csv(args.input, encoding='utf-8')
    original_len = len(df)
    if args.verbose:
        df.info()
    logger.info(f"1. Done!")
    logger.info(f"2. Dropping all the columns except the one specified: {args.column}")
    df = df.loc[:, [args.column]]
    df[args.column] = df[args.column].astype("str")
    df[args.column] = df[args.column].str.strip()
    logger.info(f"2. Done!")
    logger.info(f"3. Removing empty rows and duplicates")
    df = remove_nans(df)
    df = df.drop_duplicates()
    logger.info(f"3. {len(df)} rows left ({(1 - len(df)/original_len) * 100:.4f}% reduction)")
    if args.verbose:
        df.info()
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
    parser.add_argument("-v", "--verbose", action='store_true', help="Whether or not to write detailed dataframe info to stdout", default=True)
    
    args: ap.Namespace = parser.parse_args()
    main(args)
    