import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES_ROOT = Path('/ocean/projects/phy250048p/shared/samples')

def parse_args():
    parser = argparse.ArgumentParser(description='Transform samples for KL-NN')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input CSV file containing samples')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output CSV file to save transformed samples')
    return parser.parse_args()

def transform_samples(input_file, output_file):
    # Load samples from the input CSV file
    df = pd.read_csv(input_file)
    
    df['g1'] = -df['g1']
    df['g2'] = -df['g2']
    df['theta_int'] = df['theta_int'] - np.pi / 2

    # Save the transformed samples to the output CSV file
    df.to_csv(output_file, index=False)
    print(f'Transformed samples saved to {output_file}')

def main():
    args = parse_args()
    input_file = SAMPLES_ROOT / args.input_file
    output_file = SAMPLES_ROOT / args.output_file
    transform_samples(input_file, output_file)

if __name__ == '__main__':
    main()