import argparse
import numpy as np
import os
import subprocess
import torch
import shutil
import transformers

from datetime import datetime
from transformers import AutoModelForCausalLM

class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True

def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)


def merge_models(model1, model2, gradient_start, gradient_end):
    """
    Merge two models by blending their state_dicts.

    Args:
    - model1: The first model object to merge.
    - model2: The second model object to merge.
    - gradient_start: The start gradient value.
    - gradient_end: The end gradient value.
    """

    # No Torch gradients needed since we're only adjusting the weights and not training
    with torch.no_grad():

        # Get the state_dicts of both models
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        # Create a gradient blend ratio for both models
        blend_ratio_model2 = np.linspace(gradient_start, gradient_end, len(state_dict1))
        blend_ratio_model1 = 1 - blend_ratio_model2

        # Loop through the state_dict to merge the tensors
        for idx, key in enumerate(state_dict1.keys()):
            # Get blend ratio for the current tensor
            first_ratio = blend_ratio_model1[idx]
            second_ratio = blend_ratio_model2[idx]

            # Blend the tensors using the blend ratios
            state_dict1[key] = (first_ratio * state_dict1[key] + second_ratio * state_dict2[key])

            # Print log of blending ratios for current tensor
            print(f"{datetime.now().strftime('%H:%M:%S')} - Merging tensor {key}")
            print(str(first_ratio) + ' - ' + str(second_ratio))

        # Load the blended state_dict to the first model
        model1.load_state_dict(state_dict1)


def main(args):
    clear_console()
    print(f"{datetime.now().strftime('%H:%M:%S')} - Starting script, please wait...")

    with torch.no_grad():
        torch.set_default_dtype(torch.float32)

        # Using swap memory for the process (Unless you have 128 GB RAM...)
        device = torch.device("cpu")
        print(device)

        with NoInit():
            # Load Model 1
            print(f"{datetime.now().strftime('%H:%M:%S')} - Loading Model 1 ({args.model_path1})...")
            model1 = AutoModelForCausalLM.from_pretrained(args.model_path1)
            model1.half()
            model1 = model1.to(device)
            model1.eval()
            print(f"Model 1 Loaded. Dtype: {model1.dtype}")
    
            # Load Model 2
            print(f"{datetime.now().strftime('%H:%M:%S')} - Loading Model 2 ({args.model_path2})...")
            model2 = AutoModelForCausalLM.from_pretrained(args.model_path2)
            model2.half()
            model2 = model2.to(device)
            model2.eval()
            print(f"{datetime.now().strftime('%H:%M:%S')} -  Model 2 Loaded. Dtype: {model2.dtype}")

        # Merge the models
        print(f"{datetime.now().strftime('%H:%M:%S')} - Merging models...")
        merge_models(model1, model2, args.gradient_start, args.gradient_end)

        if args.output_model_path:
            print(f"{datetime.now().strftime('%H:%M:%S')} - Saving new model...")
            model1.save_pretrained(args.output_model_path, max_shard_size=args.max_shard_size)

            print(f"{datetime.now().strftime('%H:%M:%S')} - Saved to: {args.output_model_path}")
            print(f"{datetime.now().strftime('%H:%M:%S')} - Copying files to: {args.output_model_path}")
            files_to_copy = [
                "added_tokens.json",
                "tokenizer.model",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt"
            ]

            for filename in files_to_copy:
                src_path = os.path.join(args.model_path1, filename)
                dst_path = os.path.join(args.output_model_path, filename)
                try:
                    shutil.copy2(src_path, dst_path)
                except FileNotFoundError:
                    print(f"File {filename} not found in {args.model_path1}. Skipping.")

        print(f"{datetime.now().strftime('%H:%M:%S')} - Script Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Models')
    parser.add_argument('--model_path1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model_path2', type=str, required=True, help='Path to second model')
    parser.add_argument('--output_model_path', type=str, required=True, help='Output path for the merged model')
    parser.add_argument('--gradient_start', type=float, default=0.0, help='Starting gradient value')
    parser.add_argument('--gradient_end', type=float, default=1.00, help='Ending gradient value')
    parser.add_argument('--max_shard_size', type=str, default="2000MiB", help='Output shard size')
    
    args = parser.parse_args()
    main(args)
