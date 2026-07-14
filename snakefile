import sys, os
sys.path.append("scripts")
from protein_fold_lib import fasta_to_hash, build_alphafold_json

from Bio import SeqIO

configfile: "config.yaml"

MODE = config["mode"]

def runtime_minutes(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 60 + m + (1 if s else 0)

if MODE == "separate": 
    cfg = config["separate"]
    sp = cfg["species"]
    key = cfg["keys"]
    species_aa = cfg["species_aa"] or f'{cfg["libpath"]}/{cfg["libtype"]}/fasta/{sp}.fasta'
    with open(species_aa, "r") as fh:
        s_hash = fasta_to_hash(fh, key)
    IDS = [line.strip() for line in open(cfg["idfile"]) if line.strip()]
    TARGETS = [f"results/separate/{seqid}/{seqid}_model.cif" for seqid in IDS]


elif MODE == "pairwise_two":
    cfg = config["pairwise_two"]
    sp1, sp2 = cfg["species"].split(config["delimiter"])
    key1, key2 = cfg["keys"].split(config["delimiter"])
    species1_aa = cfg["species1_aa"] or f'{cfg["libpath"]}/{cfg["libtype"]}/fasta/{sp1}.fasta'
    species2_aa = cfg["species2_aa"] or f'{cfg["libpath"]}/{cfg["libtype"]}/fasta/{sp2}.fasta'
    with open(species1_aa, "r") as fh1:
        s1_hash = fasta_to_hash(fh1, key1)
    with open(species2_aa, "r") as fh2:
        s2_hash = fasta_to_hash(fh2, key2)
    IDS1 = [line.strip() for line in open(cfg["idfile1"]) if line.strip()]
    IDS2 = [line.strip() for line in open(cfg["idfile2"]) if line.strip()]
    PAIRS = sorted({tuple(sorted((a, b))) for a in IDS1 for b in IDS2})
    TARGETS = [f"results/pairs/{a}/{b}/{a}_{b}/{a}_{b}_model.cif" for a, b in PAIRS]


else:
    raise ValueError(f"Unknown mode: {MODE}")

rule all:
    input:
        TARGETS
        

#separate

rule build_json:
    output:
        "results/separate/{seqid}/{seqid}.json"
    run:
        build_alphafold_json(output[0], s_hash[wildcards.seqid])

rule fold_separate:
    input:
        json="results/separate/{seqid}/{seqid}.json"
    output:
        cif="results/separate/{seqid}/{seqid}_model.cif"
    params:
        final_dir="results/separate/{seqid}",
        id_string="{seqid}"
    resources:
        gpu=config["jgpu"],
        mem_mb=config.get("separate", {}).get("jmem", config["jmem"]) * 1000,
        runtime=runtime_minutes(config.get("separate", {}).get("jwalltime", config["jwalltime"])),
        cpus_per_task=config["jcpu"]
    shell:
        """
        export TMPDIR={params.final_dir}
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export TF_FORCE_UNIFIED_MEMORY=true
        export XLA_CLIENT_MEM_FRACTION=3.2
        export XLA_FLAGS="${{XLA_FLAGS}} --xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_enable_triton_gemm=false"
        mkdir -p {params.final_dir}/jax

        nvcc_location=$( {{ command -v nvcc || true; }} )
        if [[ ! -z "${{nvcc_location}}" ]]; then
          cuda_location=$(dirname $(dirname ${{nvcc_location}}))
          query_location="${{cuda_location}}/extras/demo_suite/deviceQuery"
          query_string=query_location="${{cuda_location}}/extras/demo_suite/deviceQuery"
          if [[ -x "$query_location" ]]; then
            echo $query_string >> {params.final_dir}/queryDevice.stdout
          fi
        fi

        /usr/bin/time -v -o {params.final_dir}/{params.id_string}.time -a \
          run_alphafold.py \
            --json_path {input.json} \
            --model_dir $ALPHA_HOME/models \
            --output_dir {params.final_dir} \
            --jax_compilation_cache_dir {params.final_dir}/jax \
            --flash_attention_implementation=xla \
            1>{params.final_dir}/stdout 2>{params.final_dir}/stderr
        """


#pairwise_two

rule build_pair_json:
    output:
        "results/pairs/{first}/{second}/{first}_{second}.json"
    run:
        s1 = s1_hash[wildcards.first]
        s2 = s2_hash[wildcards.second]

        build_alphafold_json(output[0], s1, seq2_record=s2)

rule fold_pairwise:
    input:
        json="results/pairs/{first}/{second}/{first}_{second}.json"
    output:
        cif="results/pairs/{first}/{second}/{first}_{second}/{first}_{second}_model.cif"
    params:
        final_dir="results/pairs/{first}/{second}",
        id_string="{first}_{second}"
    resources:
        gpu=config["jgpu"],
        mem_mb=config.get("pairwise_two", {}).get("jmem", config["jmem"]) * 1000,
        runtime=runtime_minutes(config.get("pairwise_two", {}).get("jwalltime", config["jwalltime"])),
        cpus_per_task=config["jcpu"]
    shell:
        """
        export TMPDIR={params.final_dir}
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export TF_FORCE_UNIFIED_MEMORY=true
        export XLA_CLIENT_MEM_FRACTION=3.2
        export XLA_FLAGS="${{XLA_FLAGS}} --xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_enable_triton_gemm=false"
        mkdir -p {params.final_dir}/jax

        nvcc_location=$( {{ command -v nvcc || true; }} )
        if [[ ! -z "${{nvcc_location}}" ]]; then
          cuda_location=$(dirname $(dirname ${{nvcc_location}}))
          query_location="${{cuda_location}}/extras/demo_suite/deviceQuery"
          query_string=query_location="${{cuda_location}}/extras/demo_suite/deviceQuery"
          if [[ -x "$query_location" ]]; then
            echo $query_string >> {params.final_dir}/queryDevice.stdout
          fi
        fi

        /usr/bin/time -v -o {params.final_dir}/{params.id_string}.time -a \
          run_alphafold.py \
            --json_path {input.json} \
            --model_dir $ALPHA_HOME/models \
            --output_dir {params.final_dir} \
            --jax_compilation_cache_dir {params.final_dir}/jax \
            --flash_attention_implementation=xla \
            1>{params.final_dir}/stdout 2>{params.final_dir}/stderr
        """