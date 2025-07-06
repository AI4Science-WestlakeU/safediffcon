import os
import numpy as np
import argparse

from tqdm.auto import tqdm
from datasets import Dataset



def consolidate_dataset(path, start_idx=0, end_idx=500):
    all_inputs, all_outputs, all_actions, all_targets = [], [], [], []

    # for file_name in tqdm(os.listdir(path)):
    #     if file_name.endswith('.npz'):
    for i in tqdm(range(start_idx, end_idx)):
        file_name = f"{i}.npz"
        data = np.load(os.path.join(path, file_name), allow_pickle=True)["data"].item()
        all_inputs.append(data["inputs"])
        all_outputs.append(data["outputs"])
        all_actions.append(data["actions"])
        all_targets.append(data["targets"])

    # Stack all the data in new axis
    inputs = np.stack(all_inputs, axis=0)
    outputs = np.stack(all_outputs, axis=0)
    actions = np.stack(all_actions, axis=0)
    targets = np.stack(all_targets, axis=0)
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict({
        "inputs": inputs,
        "outputs": outputs,
        "actions": actions,
        "targets": targets
    })

    return dataset


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Consolidate datasets')
    parser.add_argument('--path', type=str, default="/tokamak_data", help='Path to the dataset')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=50000, help='End index')
    
    args = parser.parse_args()
    
    dataset = consolidate_dataset(args.path, args.start_idx, args.end_idx)
    
    # save the consolidated dataset
    dataset.save_to_disk(os.path.join(args.path, "consolidated_dataset"))
    
    # if you want to load the dataset
    # dataset = datasets.load_from_disk(os.path.join(args.path, "consolidated_dataset"))
    # as pytorch format
    # dataset = dataset.with_format("torch")
    
    print(dataset)