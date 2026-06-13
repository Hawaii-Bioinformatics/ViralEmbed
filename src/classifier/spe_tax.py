from Bio import Entrez
from Bio import SeqIO
import pickle
from tqdm import tqdm 

Entrez.email = "" # enter email

def get_taxonomy_info(virus_name):
    # Search in NCBI db
    search_handle = Entrez.esearch(db="taxonomy", term=virus_name)
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    if search_results["IdList"]:
        tax_id = search_results["IdList"][0]
        fetch_handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
        data = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        taxonomy = data[0]
        tax_dict = {d['Rank']: d['ScientificName'] for d in taxonomy['LineageEx']}
        tax_dict[taxonomy['Rank']] = taxonomy['ScientificName']
        
        return {
            'superkingdom': tax_dict.get('superkingdom', 'N/A'),
            'class': tax_dict.get('class', 'N/A'),
            'order': tax_dict.get('order', 'N/A'),
            'family': tax_dict.get('family', 'N/A'),
            'genus': tax_dict.get('genus', 'N/A'),
            'species': tax_dict.get('species', 'N/A'),
        }
    else:
        return {'superkingdom': 'unclassified',
            'class': 'unclassified',
            'order': 'unclassified',
            'family': 'unclassified',
            'genus': 'unclassified',
            'species': 'unclassified',
        }

def changing_tax(saving_folder) :

    with open(f'{saving_folder}taxonomy.pkl','rb') as file : 
        taxonomy = pickle.load(file)

    corrected_dic = {}

    for t in tqdm(taxonomy) : 
        if t[0] not in corrected_dic : 
            try : 
                tax_info = get_taxonomy_info(t[0]) 
            except Exception as e : 
                tax_info = {'superkingdom': 'unclassified',
            'class': 'unclassified',
            'order': 'unclassified',
            'family': 'unclassified',
            'genus': 'unclassified',
            'species': 'unclassified'}
            corrected_dic[t[0]] = tax_info

    with open(f'{saving_folder}taxonomy_c.pkl','wb') as f2 : 
        pickle.dump(corrected_dic, f2)

changing_tax('') # destination folder
