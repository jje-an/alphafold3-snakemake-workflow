# Snakemake workflow for Alphafold3

A snakemake workflow for Alphafold3 for use on compute clusters with the Slurm workload manager.

## Requirements 

- [Snakemake](https://snakemake.readthedocs.io/en/stable/) (version 9.23.1)
- [Snakemake's Slurm executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) (version 2.7.1)
- [Alphafold3](https://github.com/google-deepmind/alphafold3) installation with model weights. (version 3.0.3)

## Configuration

Configuration options are specified in `config.yaml`, where you can specify the mode, input filepaths, and library filepaths.

### Modes

- Separate - Takes an input of a text file containing Ensembl IDs and submits individual Alphafold jobs to fold each individual protein sequence.
- Pairwise - Takes two input text files containing Ensembl IDs and submits jobs of each pair of proteins from the first and second input file. For example, if file 1 had proteins A, B, and C, and file 2 had proteins D, E, and F, there would be 9 results: pairs AD, AE, AF, BD, BE, BF, and CD, CE, CF.

## Usage

It is reccommended to run the snakemake command in a terminal multiplexer such as Tmux or screen so that the execution will not be canceled if SSH connection drops.
Do a dry run first to check for errors and ensure correct target files.

```bash
snakemake -n
```

A basic command to run the file locally is:

```bash
snakemake --cores 4 --jobs unlimited #adjust cores as needed
```

When running with the Slurm executor plugin, use the command:

```bash
snakemake --cores 4 --jobs unlimited --executor slurm --default-resources
```

Additional Slurm-related flags can be found in the [Slurm executor plugin documentation](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)

## Logs

Logs will be saved in `.snakemake/log/`. When the Slurm executor plugin is used, full error messages are located in `.snakemake/slurm_logs/`. `stderr` and `stdout` files are also saved in the `results/` folder.

## Metrics

Various helper scripts are located in `scripts` for analyzing metrics.
- `af3_summaries_to_csv.py` - Combines AF3's confidence json files from multiple outputs into one csv file.
- `plot_contact_pae_composite.py` - Creates plots emphasizing atoms with high contact probability and low PAE.
- `plot_contact_probs.py` - Creates plots of contact probabilities from full confidence metrics.
- `plot_pae_matrix.py` - Creates plots of PAE from full confidence metrics.
- `protein_fold_lib.py` - Helper functions for the Snakefile

## Output Structure

All outputs will be located in `results/'. The output structure is slightly different depending on the mode used.

### Separate

Each sample's results are in their own directory, using the same names as each line in the input file. Within this, Alphafold3 will create an additional directory with the same name containing all AF3 results. The sample directory will also contain the `.json` file used for AF3, a `.time` file, and `stderr`/`stdout` files.

### Pairwise

Each sample pair's directory is formatted as: `result/pairs/{first_id}/{second_id}/`. This directory contains AF3's generated directory, the `.json` file, `.time` file, and `stderr`/`stdout`. For example, for an input of two files with proteins A, B, and C, D respectively, the directory would look like:

```
results
├── pairs
│   ├── A
│   │   ├── C
|   |   |   ├── A_C # Alphafold3's result directory
|   |   |   |   └── ...
|   |   |   ├── A_C.json
|   |   |   ├── A_C.time
|   |   |   ├── stderr
|   |   |   └── stdout
│   │   └── D
|   |       └── ...
│   └── B
│       ├── C
|       |   ├── ...
│       └── D
|           └── ...
└── separate
```