import random
import argparse
import json
from typing import Optional

import statistics


def generate_bimodal_distribution(
        max_value: int, 
        count: int, 
        min_median: int,
        max_median: int,
        generator: Optional[random.Random] = None,
    ) -> dict:
    assert count >= 1
    assert min_median <= max_median
    
    if generator is None:
        generator = random
        
    target_median = generator.randint(min_median, max_median)
    
    lower_half_count = count // 2
    upper_half_count = count - lower_half_count
    
    values = []
    
    for _ in range(lower_half_count):
        value = generator.randint(0, target_median - 1)
        values.append(value)
        
    for _ in range(upper_half_count):
        value = generator.randint(target_median, max_value)
        values.append(value)
    
    generator.shuffle(values)
    
    output_data = {
        "parameters": {
            "max_value": max_value,
            "count": count,
            "target_median": target_median,
        },
        "values": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
    }
    
    return output_data

def generate_uniform_distribution(
        max_value: int, 
        count: int, 
        generator: Optional[random.Random] = None,
    ) -> dict:
    if generator is None:
        generator = random
    
    values = [generator.randint(0, max_value) for _ in range(count)]

    output_data = {
        "parameters": {
            "max_value": max_value,
            "count": count,
        },
        "values": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
    }
    
    return output_data



import argparse
import glob
import json
import os
import re
import random

import datasets
import torch
import tqdm

MEDIAN_DATA_DIR = os.path.expanduser("~/verl/data/median")


class MedianDatasetVerl(torch.utils.data.Dataset):
    def __init__(
        self, 
        num_samples: int, 
        list_length: int,
        max_seq_length: int,
        min_median: int,
        max_median: int,
        max_value: int,
        seed: int,
        is_evaluation: bool = False,
    ):
        self.num_samples = num_samples
        self.list_length = list_length
        self.max_seq_length = max_seq_length
        self.min_median = min_median
        self.max_median = max_median
        self.max_value = max_value
        self.seed = seed
        self.is_evaluation = is_evaluation

        self.distribution_type = "bimodal"
        self.rng = random.Random(seed)
        self.seeds = [self.rng.randint(0, 2**32-1) for _ in range(num_samples)]

        self.THINK_TOKEN = "think"
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        random.seed(self.seeds[idx])
        sample_data = generate_bimodal_distribution(
            max_value=self.max_value,
            count=self.list_length,
            min_median=self.min_median,
            max_median=self.max_median,
        )
        
        # Create the instruction
        prompt = f"Given the following data, consisting of {self.list_length} numbers, you will be asked to compute the median. Data: {sample_data['values']}. What is the median of these numbers? Feel free to think step-by-step. After reasoning, please respond with the median only (a single number)."
        instruction_following = 'Let\'s think step by step and output the final answer after "####".'
        
        answer = str(sample_data['median'])

        return {
            "data_source": "median", # Just use MATH reward
            "prompt": [
                {"role": "user", "content": prompt + " " + instruction_following}
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                'index': idx
            }
        }
    
    def to_hf_dataset(self) -> datasets.Dataset:
        return datasets.Dataset.from_list([self[i] for i in tqdm.trange(len(self), leave=False, desc='Converting to HF dataset')])


def generate_verl_datasets(args: argparse.Namespace, data_dir: str):
    # Calculate max sequence length
    # TODO: find a better way to calculate this.
    max_seq_length = 256 + args.list_length * 5
    
    # Create on-the-fly datasets for training and evaluation
    train_dataset = MedianDatasetVerl(
        num_samples=args.train_samples,
        list_length=args.list_length,
        max_seq_length=max_seq_length,
        seed=args.seed,
        min_median=args.min_median,
        max_median=args.max_median,
        max_value=args.max_value,
        is_evaluation=False,
    ).to_hf_dataset()
    
    eval_dataset = MedianDatasetVerl(
        num_samples=args.eval_samples,
        list_length=args.list_length,
        max_seq_length=max_seq_length,
        seed=args.seed + 1,  # Use different seed for eval
        min_median=args.min_median,
        max_median=args.max_median,
        max_value=args.max_value,
        is_evaluation=True,
    ).to_hf_dataset()
    
    eval_datasets = {
        "eval": eval_dataset,
    }

    print(f"[main] Saving data to {data_dir} (*.parquet)")

    train_file_path = os.path.join(data_dir, 'train.parquet')
    eval_dataset_paths = []
    for name, dataset in eval_datasets.items():
        file_path = os.path.join(data_dir, "eval", f'{name}.parquet')
        dataset.to_parquet(file_path)
        eval_dataset_paths.append(file_path)
        print(f"[main] Wrote {name} dataset of length {len(dataset)} to {file_path}")

    train_dataset.to_parquet(train_file_path)
    print(f"[main] Wrote train dataset of length {len(train_dataset)} to {train_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLaMA model for median computation')
    
    parser.add_argument('--list_length', type=int, default=35,
                      help='Length of number lists in training data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Training configuration
    parser.add_argument('--train_samples', type=int, default=7_500,
                      help='Number of training samples to generate')
    parser.add_argument('--eval_samples', type=int, default=2_500,
                      help='Number of evaluation samples to generate')
    parser.add_argument('--min_median', type=int, default=100,
                      help='Minimum median value')
    parser.add_argument('--max_median', type=int, default=400,
                      help='Maximum median value')
    parser.add_argument('--max_value', type=int, default=500,
                      help='Maximum value for numbers in the distribution')
    args = parser.parse_args()

    generate_verl_datasets(args, os.path.join(MEDIAN_DATA_DIR, str(args.list_length)))

    
if __name__ == "__main__": main()
