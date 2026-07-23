import csv
import glob
import json
import os
import sys
import statistics


def af3_summaries_to_csv(input_dir=None, output_csv="af3_summary.csv"):
    if input_dir is None:
        input_dir = os.getcwd()
    rows = []

    # three directories deep so it will ignore per-sample confidences
    # and so you can provide a parent directory with all jobs

    summary_files = sorted(glob.glob(f"{input_dir}/*/*/*/*summary_confidences.json"))
    confidence_files = sorted(f for f in glob.glob(f"{input_dir}/*/*/*/*confidences.json") 
                              if not f.endswith("summary_confidences.json"))

    for summary, full in zip(summary_files, confidence_files):

        with open(summary) as f:
            summary_data = json.load(f)
        with open(full) as f:
            full_data = json.load(f)
        
        # this order of columns is so that it matches the esm summaries as best it can
        row = {
            "job_id": summary.split("/")[-1].split("summary_confidences.json")[0].rstrip("_-"),
            "iptm": summary_data.get("iptm"),
            "ptm": summary_data.get("ptm")
        }
        
        
        pair_iptm = summary_data["chain_pair_iptm"]
        for i, row_vals in enumerate(pair_iptm, start=1):
            for j, val in enumerate(row_vals, start=1):
                row[f"chain_pair_iptm_{i}_{j}"] = val
        
        plddt = [x for x in full_data.get("atom_plddts") if x is not None]
        mean = statistics.mean(plddt)
        row["plddt_mean"] = round(mean, 2)

        pair_pae = summary_data["chain_pair_pae_min"]
        for i, row_vals in enumerate(pair_pae, start=1):
            for j, val in enumerate(row_vals, start=1):
                row[f"chain_pair_pae_min_{i}_{j}"] = val

        for i, val in enumerate(summary_data["chain_iptm"], start=1):
            row[f"chain_iptm_{i}"] = val

        for i, val in enumerate(summary_data["chain_ptm"], start=1):
            row[f"chain_ptm_{i}"] = val
        row.update(
            {
                "fraction_disordered": summary_data.get("fraction_disordered"),
                "has_clash": summary_data.get("has_clash"),
                "ranking_score": summary_data.get("ranking_score")
            }
        )
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
