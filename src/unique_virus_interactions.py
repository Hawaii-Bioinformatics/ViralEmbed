from tqdm import tqdm


def full_interactions(input_file, output_file, virus_id) : 
    count = 0
    with open(input_file, 'r') as file:
        with open(output_file, 'w') as output:
            for line in tqdm(file):
                if line.startswith(virus_id):
                    output.write(line)
                    count+=1

    print(f"File has been written with selected lines : {count} interactions.")