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

def get_sequence(fa_path, key, seq_id, out_path):
    with open(fa_path, "r") as fh:
        s_hash = fasta_to_hash(fh, key)

    s = s_hash[seq_id]
    build_alphafold_json(out_path, s)


def get_sequence_pair(fa1_path, fa2_path, key1, key2, id1, id2, out_path):
    with open(fa1_path, "r") as fh1:
        s1_hash = fasta_to_hash(fh1, key1)
    with open(fa2_path, "r") as fh2:
        s2_hash = fasta_to_hash(fh2, key2)

    s1 = s1_hash[id1]
    s2 = s2_hash[id2]

    build_alphafold_json(out_path, s1, seq2_record=s2)

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