from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def proteins_fasta(input_file, output_file, virus_id) : 
    selected_sequences = []

    i = 0

    # Ouverture et lecture du fichier FASTA
    with open(input_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record.id.startswith(virus_id):
                new_id = record.id.split()[0] + f"_{i}"  # Ajouter _i à la fin de l'identifiant
                record.id = new_id
                record.description = ''
                selected_sequences.append(record)
                i+=1

    with open(output_file, "w") as output_handle:
        SeqIO.write(selected_sequences, output_handle, "fasta")

    print(f"{len(selected_sequences)} sequences have been written to {output_file}")


def virus_fasta(input_file, output_file, virus_id) : 
    assembled_sequence = ''

    with open(input_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            assembled_sequence += str(record.seq)

    print(f"The virus has a length of {len(Seq(assembled_sequence))} amino acids")

    new_record = SeqRecord(
        Seq(assembled_sequence),  # Assembler la séquence
        id=virus_id,               # Nouvel identifiant
        description=""            # Description vide
    )

    with open(output_file, "w") as output_handle:
        SeqIO.write(new_record, output_handle, "fasta")

    print(f"Sequence has been assembled and written to {output_file}")