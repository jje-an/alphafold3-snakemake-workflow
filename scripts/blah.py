import re
import json
from Bio import SeqIO


def fasta_to_hash(file_handle, key):
    output = {}

    for record in SeqIO.parse(file_handle, "fasta"):
        if key:
            seqid = record.description
        else:
            seqid = record.id
        short_id = seqid
        if key:
            match = re.search(rf"{key}\s*[:|=](\S+)", seqid)
        else:
           match = None
        if (match):
            short_id = match.group(1) 
            ## Get rid of terminal ID numbers (assume it doesn't pass 99)
            short_id = re.sub(r"\.\d{1,2}$", "", short_id) 
        output[short_id] = record

    return output

#Search for one sequence by short_id out of reference and append in separate file
def extract_sequence(fasta_path, seq_id, key, out_path):
    with open(fasta_path, "r") as fh:
        seq_hash = fasta_to_hash(fh, key)
    if seq_id not in seq_hash:
        raise KeyError(f"'{seq_id}' not found in {fasta_path} (key={key!r})")
    record = seq_hash[seq_id]
    with open(out_path, "w") as out_fh:
        SeqIO.write(record, out_fh, "fasta")
        
def build_alphafold_json(out_path, seq1_record, seq2_record=None, molecule_type="protein"):
    if seq2_record:
        datum = {
            "name": f"{seq1_record.id}_{seq2_record.id}",
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
            "sequences": [
                {molecule_type: {"id": ["A"], "sequence": str(seq1_record.seq)}},
                {molecule_type: {"id": ["B"], "sequence": str(seq2_record.seq)}},
            ],
        }
    else:
        datum = {
            "name": f"{seq1_record.id}",
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
            "sequences": [
                {molecule_type: {"id": ["A"], "sequence": str(seq1_record.seq)}}
            ],
        }
    with open(out_path, "w") as f:
        json.dump(datum, f, indent=2)