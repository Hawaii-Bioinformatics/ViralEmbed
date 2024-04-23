import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import statistics

def read_interaction_file(filepath):
    interactions = {}
    scores = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            key = (parts[1], parts[3])  
            score = float(parts[4])
            interactions[key] = score
            scores.append(score)
    return interactions, scores

def read_full_interaction_file(filepath):
    interactions = {}
    exp_scores = []
    mining_scores = []
    scores = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            key = (parts[1], parts[3])  
            exp_score = int(parts[-7])+int(parts[-6])
            mining_score = int(parts[-3])+int(parts[-2])
            score = int(parts[-1])
            interactions[key] = (exp_score, mining_score, score)
            exp_scores.append(exp_score)
            mining_scores.append(mining_score)
            scores.append(score)
    return interactions, exp_scores, mining_scores, scores

def binary_labels(interactions, threshold):
    return {key: 1 if score > threshold else 0 for key, score in interactions.items()}

def binary_full_labels(interactions, threshold, i):
    return {key: 1 if score[i] > threshold else 0 for key, score in interactions.items()}

def classification_metrics(results_data, results_scores, interaction_data_full, score_type, distances_data, distances_scores, string_tr, model_tr) : 
    # Predicted interactions : 
    threshold = model_tr
    #threshold = 0.01
    results_labels = binary_labels(results_data, threshold)

    # String interaction scores : 
    i=2
    if score_type == "experimental_score" : 
        i=0
    elif score_type == "mining_score" : 
        i=1    
    elif score_type == "complete_score" :
        i=2
    elif score_type == "sizes" :
        scores_labels = binary_labels(distances_data, statistics.median(distances_scores)/2)
        common_keys = results_data.keys() & distances_data.keys()
        
        cm = confusion_matrix([scores_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
        true_negative, false_positive, false_negative, true_positive = cm.ravel()

        accuracy = accuracy_score([scores_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
        precision = precision_score([scores_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
        recall = recall_score([scores_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
        f1 = f1_score([scores_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])

        print(f"True Positives: {true_positive}")
        print(f"False Positives: {false_positive}")
        print(f"True Negatives: {true_negative}")
        print(f"False Negatives: {false_negative}")

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

        return


    interaction_labels = binary_full_labels(interaction_data_full, string_tr, i)

    common_keys = results_data.keys() & interaction_data_full.keys()

    cm = confusion_matrix([interaction_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
    true_negative, false_positive, false_negative, true_positive = cm.ravel()

    accuracy = accuracy_score([interaction_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
    precision = precision_score([interaction_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
    recall = recall_score([interaction_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])
    f1 = f1_score([interaction_labels[key] for key in common_keys], [results_labels[key] for key in common_keys])

    print(f"True Positives: {true_positive}")
    print(f"False Positives: {false_positive}")
    print(f"True Negatives: {true_negative}")
    print(f"False Negatives: {false_negative}")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

def pearson_correlation(results_data, interaction_data_full, score_type, distances_data, distances_scores) : 
    i=2
    if score_type == "experimental_score" : 
        i=0
    elif score_type == "mining_score" : 
        i=1    
    elif score_type == "complete_score" :
        i=2
    elif score_type == "sizes" : 
        common_keys = results_data.keys() & distances_data.keys()
        results_scores = [results_data[key] for key in common_keys]
        correlation_coefficient = np.corrcoef(results_scores, distances_scores)[0, 1]
        print(f"Pearson correlation coefficient between sequence sizes and predicted contacts : {correlation_coefficient:.3f}")
        return

    common_keys = results_data.keys() & interaction_data_full.keys()
    results_scores = [results_data[key] for key in common_keys]
    interaction_scores = [interaction_data_full[key][i] for key in common_keys]
    correlation_coefficient = np.corrcoef(results_scores, interaction_scores)[0, 1]

    print(f"Pearson correlation coefficient between String and predicted contacts : {correlation_coefficient:.3f}")

    return

def all_metrics(results_data, results_scores, interaction_data_full, score_type, distances_data, distances_scores, string_tr, model_tr) : 

    if score_type == "experimental_score" : 
        print("Score String déterminé par expériences/biochimie")
    elif score_type == "mining_score" : 
        print("Score String déterminé par textmining")
    elif score_type == "complete_score" :
        print("Score String complet")
    elif score_type == "distances" : 
        print("Protein sizes")

    pearson_correlation(results_data, interaction_data_full, score_type, distances_data, distances_scores)
    classification_metrics(results_data, results_scores, interaction_data_full, score_type,distances_data, distances_scores, string_tr, model_tr)
    
    return

#results_path = '/home/thibaut/contacts/influenza/results_freq.txt'
#interactions_path = '/home/thibaut/String/Influenza/Influenza_A_interactions.txt'
#interactions_full_path = '/home/thibaut/String/viruses/virus_interactions_full.txt'
#distances_path = '/home/thibaut/String/Influenza/Influenza_A_sizes.txt'
#results_data, results_scores = read_interaction_file(results_path)
#interaction_data, interaction_scores = read_interaction_file(interactions_path)
#interaction_data_full, interaction_mining_scores , interaction_exp_scores, interaction_scores_full = read_full_interaction_file(interactions_full_path)
#distances_data, distances_scores = read_interaction_file(distances_path)
#string_tr = 300
#model_tr = statistics.median(results_scores)/4

"""
matches = []
for key in results_data:
    if key in interaction_data:
        match_score = (results_data[key], interaction_data[key])
        matches.append(match_score)

        # print(f"Match found for {key}: Results Score = {results_data[key]}, Interaction Score = {interaction_data[key]}")

# print(f"{len(matches)} matches found")
"""
"""
# Tracer les distributions des scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.hist(results_scores, bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Predicted contacts')
plt.xlabel('Count')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(interaction_scores, bins=30, alpha=0.7, color='green')
plt.title('Distribution of Interaction Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.hist(distances_scores, bins=30, alpha=0.7, color='orange')
plt.title('Distribution of Distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.tight_layout() 

"""
# plt.savefig("/home/thibaut/contacts/influenza/distributions.png")

# Total scores : 
# all_metrics(results_data, results_scores, interaction_data_full, "complete_score", distances_data, distances_scores, string_tr, model_tr)

# Experimental scores : 
#all_metrics(results_data, results_scores, interaction_data_full, score_type = "experimental_score", distances_data, distances_scores, string_tr)

# Textmining_scores : 
#all_metrics(results_data, results_scores, interaction_data_full, score_type = "mining_score", distances_data, distances_scores, string_tr)

# Distances : 
# all_metrics(results_data, results_scores, interaction_data_full, "sizes", distances_data, distances_scores, string_tr)
