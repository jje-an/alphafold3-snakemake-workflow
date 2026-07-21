import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import glob

def plot_pae_matrix(input_dir=None, output_dir=None):
    if input_dir is None:
        input_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(f"{input_dir}/*confidences.json") 
                   if not f.endswith("summary_confidences.json"))
    
    
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        
        pae = data.get("pae")
        
        pae_np = np.array(pae, dtype=float)

        job_id = file.split("/")[-1].split("confidences.json")[0].rstrip("_")
        
        plt.imshow(pae_np, cmap="summer")
        plt.title(f'PAE Matrix for {job_id}')
        plt.xlabel("Scored Residue", labelpad=20)
        plt.ylabel("Aligned Residue", labelpad=20)
        plt.colorbar(label="PAE (Angstroms)")

        
        #separate graph based on chain
        token_chain_ids = data.get("token_chain_ids")
        unique_token_chain_ids = sorted(set(token_chain_ids))
        tokens_per_chain = []
        for i in range(len(unique_token_chain_ids)):
            tokens_per_chain.append(token_chain_ids.count(unique_token_chain_ids[i]))

        for i in range(1, len(unique_token_chain_ids)):
            
            # .5 centers line on boundary of token
            plt.axvline(sum(tokens_per_chain[0:i]) - .5, color='black', linewidth=.8)
            plt.axhline(sum(tokens_per_chain[0:i]) - .5, color='black', linewidth=.8)

        # boundary index where each chain's tokens start/end
        chain_starts = [0]
        for count in tokens_per_chain[:-1]:
            chain_starts.append(chain_starts[-1] + count)
        chain_ends = [start + count for start, count in zip(chain_starts, tokens_per_chain)]

        plt.tick_params(labelbottom=False, labelleft=False) 

        # chain name labels
        for start, end, chain_name in zip(chain_starts, chain_ends, unique_token_chain_ids):
            midpoint = (start + end) / 2 - 0.5
            plt.text(midpoint, len(token_chain_ids) + 2, f"Chain {chain_name}",
                    ha='center', va='top', fontsize=10)
            plt.text(-2, midpoint, f"Chain {chain_name}",
                    ha='right', va='center', fontsize=10, rotation=90)

        
        out_path = f"{output_dir}/{job_id}_pae_matrix.png"
        plt.savefig(out_path)
        plt.close()
    
    print(f'{len(files)} plots saved in {output_dir}')

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_pae_matrix(input_dir, output_dir)

