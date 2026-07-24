import numpy as np
import math
import os
import glob
import json
import sys
import matplotlib.pyplot as plt

# creates plot emphasizing atoms that the model thinks are in contact regions and also has a low PAE.

def plot_contact_pae_composite(input_dir=None, output_dir=None):
    if input_dir is None:
        input_dir = os.getcwd()
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(f for f in glob.glob(f"{input_dir}/*/*/*/*confidences.json") 
                   if not f.endswith("summary_confidences.json"))
    
    for file in files:
        with open(file) as f:
            data = json.load(f)

        pae = data.get("pae")
        pae_np = np.array(pae, dtype=float)

        contact_probs = data.get("contact_probs")
        contact_probs_np = np.array(contact_probs, dtype=float)

        pae_np_normalized = pae_np / np.max(pae_np)
        contact_probs_flipped = np.ones(np.shape(contact_probs_np)) - contact_probs_np

        job_id = file.split("/")[-1].split("confidences.json")[0].rstrip("_")
        
        contact_pae_composite = -1 * np.log10(pae_np_normalized * (contact_probs_flipped + 10**-16))

        plt.imshow(contact_pae_composite, cmap="summer")
        plt.title(f'High Contact Probability and Low PAE for {job_id}')
        plt.colorbar(label="-log10(Normalized PAE * Reverse Contact Probability)")

        out_path = f"{output_dir}/{job_id}_contact_pae_composite.png"
        plt.savefig(out_path)
        plt.close()
    print(f'{len(files)} plots saved in {output_dir}')


if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_contact_pae_composite(input_dir, output_dir)
