import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import glob

def plot_contact_probs(input_dir=None, output_dir=None):
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

        contact_probs = data.get("contact_probs")
        contact_probs_np = np.array(contact_probs, dtype=float)

        job_id = file.split("/")[-1].split("confidences.json")[0].rstrip("_")
        
        plt.imshow(contact_probs, cmap="summer")
        plt.title(f'Contact Probabilities for {job_id}')
        plt.colorbar(label="Probability")

        out_path = f"{output_dir}/{job_id}_contact_prob.png"
        plt.savefig(out_path)
        plt.close()
    
    print(f'{len(files)} plots saved in {output_dir}')

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_contact_probs(input_dir, output_dir)

