# %% [markdown]
# # bioinformatics glossary
# 
# 
# biopython
# entrez
# ncbi
# kmers



# %%
# baixar os arquivos
!gdown 18HITjQAuKQhVADfJliyOFLxj8hUa-AC2
!gdown 1jEaCQ97OfEphPlKVitcFdC39Zol7NPFD
!gdown 1_vBi9LQgF545MZTunw_zd0S1duEBwwee

# %%
from Bio import SeqIO

seq = list(SeqIO.parse("seqdump.txt", "fasta"))
seq

# %%
# parse uniprot.xml

seq_uniprot = list(SeqIO.parse("uniprot.xml", "uniprot-xml"))






# %%
dir(seq_uniprot[0])

for x in seq_uniprot:
    print("--------------------------------")
    # print(x.id)
    # print(x.name)
    # print(x.description)
    # print(x.seq)
    # print(x.features)
    print(x.annotations["taxonomy"])
    # print(x.dbxrefs)







# %%



