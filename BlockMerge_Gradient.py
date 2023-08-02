import argparse
import numpy as np
import os
import subprocess
import torch
import shutil

from datetime import datetime
from transformers import AutoModelForCausalLM


def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)


def merge_models(model1, model2, gradient_start, gradient_end, layer_start, layer_end):
    """
    Merge two models by blending their layers.
    
    Args:
    - model1: The first model object to merge.
    - model2: The second model object to merge.
    - gradient_start: The start gradient value.
    - gradient_end: The end gradient value.
    - layer_start: The start layer for merging.
    - layer_end: The end layer for merging.
    """
    
    # No Torch gradients needed since we're only adjusting the weights and not training
    with torch.no_grad():
        
        # Determine the total number of hidden layers
        num_layers = model1.config.num_hidden_layers

        # Ensure the layer_end value doesn't exceed the number of layers in the model
        if layer_end > num_layers:
            layer_end = num_layers

        # Calculate the number of steps for blending the layers
        num_steps = layer_end - layer_start
        
        # Create a gradient blend ratio for both models
        blend_ratio_model2 = np.linspace(gradient_start, gradient_end, num_steps)
        blend_ratio_model1 = 1 - blend_ratio_model2

        # Loop through the specified range of layers to merge them
        for idx, i in enumerate(range(layer_start, layer_end)):
            # Get blend ratio for the current layer
            first_ratio = blend_ratio_model1[idx]
            second_ratio = blend_ratio_model2[idx]

            # Extract state dictionary for current layer from both models
            merged_layer = (model1.model.layers[i].state_dict(), model2.model.layers[i].state_dict())

            # Iterate through the state dictionary and blend the parameters using the blend ratios
            for key in merged_layer[0].keys():
                merged_layer[0][key] = (first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key])

            # Load the blended parameters to the first model
            model1.model.layers[i].load_state_dict(merged_layer[0])

            # Print log of blending ratios for current layer
            print(f"{datetime.now().strftime('%H:%M:%S')} - Merging layer {i}")
            print(str(first_ratio) + ' - ' + str(second_ratio))


def main(args):
    clear_console()
    print(f"{datetime.now().strftime('%H:%M:%S')} - Starting script, please wait...")

    with torch.no_grad():
        torch.set_default_dtype(torch.float32)

        # Using swap memory for the process (Unless you have 128 GB RAM...)
        device = torch.device("cpu")
        print(device)

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
        merge_models(model1, model2, args.gradient_start, args.gradient_end, args.layer_start, args.layer_end)

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
    parser.add_argument('--layer_start', type=int, default=0, help='Start layer for merging')
    parser.add_argument('--layer_end', type=int, default=99, help='End layer for merging')
    parser.add_argument('--max_shard_size', type=str, default="2000MiB", help='Output shard size')
    
    args = parser.parse_args()
    main(args)
