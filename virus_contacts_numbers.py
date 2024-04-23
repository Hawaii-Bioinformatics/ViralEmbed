import torch
import os

# Définir le chemin du dossier contenant les fichiers .pt
directory_path = '/home/thibaut/contacts/influenza/sequence_0_mutual_information'

# Définir la liste de correspondance
id_map = {
    0: "HEMA_I34A1", 
    1: "M1_I34A1",
    2: "M2_I34A1", 
    3: "NCAP_I34A1", 
    4: "NEP_I34A1", 
    5: "NRAM_I34A1", 
    6: "NS1_I34A1", 
    7: "PAX_I34A1", 
    8: "PA_I34A1", 
    9: "PB1F2_I34A1", 
    10: "PB2_I34A1", 
    11: "RDRP_I34A1"
}

def count_contacts(output_file_path, directory_path, id_map, virus_id, freq_factor) : 
    list_tensors = []
    with open(output_file_path, 'w') as output_file:
        for filename in os.listdir(directory_path):
            if filename.endswith('.pt'):
                tensor_path = os.path.join(directory_path, filename)
                tensor = torch.load(tensor_path)
                list_tensors.append(tensor)
                sigmoid_tensor = torch.sigmoid(tensor)
                

                if torch.isnan(sigmoid_tensor).any():
                    print("NaN detected in sigmoid_tensor")
                    sigmoid_tensor[torch.isnan(sigmoid_tensor)] = 0  # Remplacement des NaN par 0
            
                # Affichage du tenseur sigmoid pour vérifier les résultats
                # print(sigmoid_tensor)
                
                # Conversion en entier après vérification de NaN
                try:
                    count = int(sigmoid_tensor.sum().item())
                except ValueError as e:
                    print(f"Error converting to int: {e}")
                    continue  

                count = int((sigmoid_tensor).sum().item())
                count = freq_factor*count / (tensor.size(0)**2) # If using frequence

                parts = filename.split('_')
                index1 = int(parts[0])
                index2 = int(parts[1])

                result_line = f"{virus_id}\t{id_map[index1]:<15}\t{virus_id}\t{id_map[index2]:<15}\t{count}\n"
                output_file.write(result_line)

        

    print(f"Results have been written to {output_file_path}")
    
    return(list_tensors, os.listdir(directory_path))

#output_file_path = "/home/thibaut/contacts/influenza/results_freq.txt"
#virus_id = "11320"
#count_contacts(output_file_path, directory_path, id_map, virus_id)