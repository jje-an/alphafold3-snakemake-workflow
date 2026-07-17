import csv
import glob
import json
import os
import sys


def af3_summaries_to_csv(input_dir=None, output_csv="af3_summary.csv"):
    if input_dir is None:
        input_dir = os.getcwd()
    rows = []
    files = sorted(glob.glob(f"{input_dir}/*/*/*/*summary_confidences.json"))
    
    for path in files:

        with open(path) as f:
            data = json.load(f)
        

        row = {
            "job_id": path.split("/")[-1].split("summary_confidences.json")[0].rstrip("_-"),
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
            "iptm": data.get("iptm"),
            "ptm": data.get("ptm"),
            "ranking_score": data.get("ranking_score"),
        }
        
        for i, val in enumerate(data["chain_iptm"], start=1):
            row[f"chain_iptm_{i}"] = val

        pair_iptm = data["chain_pair_iptm"]
        for i, row_vals in enumerate(pair_iptm, start=1):
            for j, val in enumerate(row_vals, start=1):
                row[f"chain_pair_iptm_{i}_{j}"] = val

        pair_pae = data["chain_pair_pae_min"]
        for i, row_vals in enumerate(pair_pae, start=1):
            for j, val in enumerate(row_vals, start=1):
                row[f"chain_pair_pae_min_{i}_{j}"] = val
        
        for i, val in enumerate(data["chain_ptm"], start=1):
            row[f"chain_ptm_{i}"] = val

        rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "af3_summary.csv"
    af3_summaries_to_csv(input_dir, output_csv)
