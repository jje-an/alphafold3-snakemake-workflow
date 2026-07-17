from pymol import cmd, stored
import pandas
import statistics
from io import StringIO
from Bio import SeqIO

def distance_matrix(chain_one="A", chain_two="B", output="output.csv"):
    '''
Iterate over every molecule in each of two chains and make a nxm matrix of their distances.

    '''
    sel_chain_one = cmd.select(f"//{chain_one}")
    fasta_chain_one = cmd.get_fastastr('sele')
    chain_one_io = StringIO(fasta_chain_one)
    chain_one_records = list(SeqIO.parse(chain_one_io, "fasta"))
    seq_chain_one = chain_one_records[0].seq
    len_chain_one = len(seq_chain_one)
    sel_chain_two = cmd.select(f"//{chain_two}")
    fasta_chain_two = cmd.get_fastastr('sele')
    chain_two_io = StringIO(fasta_chain_two)
    chain_two_records = list(SeqIO.parse(chain_two_io, "fasta"))
    seq_chain_two = chain_two_records[0].seq
    len_chain_two = len(seq_chain_two)
    r_one = range(len_chain_one)
    r_two = range(len_chain_two)
    selection_counter = 0
    dist_df = pandas.DataFrame(index=r_one, columns=r_two)
    for aa_one in r_one:
        print(f"Working on {aa_one} from chain A.")
        for aa_two in r_two:
            selection_counter = selection_counter + 1
            ## print(f"Working on {chain_one} residue {aa_one} vs {chain_two} residue {aa_two}")
            aa_onep = aa_one + 1
            aa_twop = aa_two + 1
            ## I am not sure if selecting an amino acid and asking for the distance will result in a single distance or a vector or what...
            ## The following creates an object in pymol which needs to be cleaned up.
            pair_distances = cmd.distance("tmp_obj", f"(//{chain_one}/{chain_one}/{aa_onep}/C*)", f"(//{chain_two}/{chain_two}/{aa_twop}/C*)", cutoff = 999.9, mode = 4)
            ## pair_dist = statistics.mean(pair_distances)
            ## The following requires one specify the atoms and will not allow wildcards
            ## pair_distances = cmd.get_distance(f"//{chain_one}/{chain_one}/{aa_onep}/C", f"//{chain_two}/{chain_two}/{aa_twop}/C*")
            ##print(pair_distances)
            dist_df.loc[aa_one, aa_two] = pair_distances
            ## Delete the created object from pymol's list of stuff to examine.
            deleted = cmd.delete("tmp_obj")
    dist_df.to_csv(output)
    return(dist_df)

cmd.extend("distance_matrix", distance_matrix);
