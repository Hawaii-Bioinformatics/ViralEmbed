import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv

tax_classes = ['superkingdom', 'class', 'genus', 'species']

saving_folder = '' # Folder with saved taxonomy pkl files (from spe_tax.py)
tax_folder = '' # Folder with saved taxonomy pkl files (from embeddings_and_tax.py)

with open(f'{tax_folder}taxes.pkl','rb') as file : 
    taxonomy = pickle.load(file)

# with open(f'{saving_folder}taxonomy_c.pkl','rb') as file : 
#     taxonomy_c = pickle.load(file)


tensor = torch.load(f'').squeeze(1) # Enter embeddings path
print(tensor.shape)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def data_loading(index, taxonomy, tensor, limit) :

    taxons = []

    # for t in taxonomy : 
    #     if t[1] is not None and 'Lineage' in t[1] : 
    #         try :
    #             taxons.append(t[1]['Lineage'].split(';')[index])
    #         except Exception : 
    #             taxons.append(t[1]['Lineage'].split(';')[-1])
    #     else : 
    #         taxons.append('unclassified')

    for t in taxonomy : 
        try : 
            tax_dic = taxonomy[t]
            label = tax_dic[tax_classes[index]]
            if label == 'N/A' : 
                taxons.append('unclassified')
            else : 
                taxons.append(label)
        except Exception : 
            taxons.append('unclassified')

    raw_labels = np.array(taxons)

    tensor = tensor.cpu()
    X = tensor.numpy()
    #print(Counter(raw_labels))
    class_counts = Counter(raw_labels)
    print(class_counts)
    #valid_classes = {cls for cls, count in class_counts.items() if count > 18 and 'unclassified' not in cls}
    top_20_elements = class_counts.most_common(limit)
    top_20_set = set([element for element, frequency in top_20_elements if 'unclassified' not in element])
    valid_classes = top_20_set
    #print(valid_classes)

    selection = False
    if selection : 
        filtered_data = [data for data, label, tax in zip(X, raw_labels, taxonomy) if label in valid_classes and tax[1][0] == 'Viruses']
        labels = [label for label, tax in zip(raw_labels, taxonomy) if label in valid_classes and tax[1][0] == 'Viruses']

    else : 
        filtered_data = [data for data, label in zip(X, raw_labels) if label in valid_classes ]
        labels = [label for label in raw_labels if label in valid_classes]

    #print(Counter(labels))
    num_classes = len(list(set(labels)))
    print(num_classes)
    
    filtered_data = np.array(filtered_data).astype(np.float32)
    labels = np.array(labels)
    print(filtered_data.shape)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(filtered_data, y, test_size=0.5, random_state=42)

    # Train / Test loaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=320, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=320, shuffle=True)

    #print(f'Training set size = {len(train_loader)} // Test set size = {len(test_loader)}')

    return X_train, X_test, y_train, y_test, train_loader, test_loader, num_classes, label_encoder

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, intermedita_size, intermedita_size_2 = 5120):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, intermedita_size)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(intermedita_size, num_classes)
        # self.relu2 = nn.LeakyReLU()
        # self.fc3 = nn.Linear(intermedita_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

single = False
if single :
    model = SimpleNN(input_size=1280, num_classes=16, intermedita_size = 2560)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 500

    model_save_path = saving_folder+'classifier_nn.pth'
    optimizer_save_path = saving_folder+'classifier_nn_optimizer.pth'


def train(train_loader, num_epochs, model_save_path, optimizer_save_path, index) : 
    loss_epochs = []
    epochs = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # if epoch%10 == 0 : 
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        epochs.append(epoch)
        loss_epochs.append(running_loss/len(train_loader))

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)

    print(f'Model saved to {model_save_path}')
    print(f'Optimizer saved to {optimizer_save_path}')

    plt.figure()
    plt.plot(epochs, loss_epochs,linewidth=1, color = 'dodgerblue')
    plt.title('Classifier training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.ylim(bottom=0)
    plt.savefig(f'{saving_folder}tr_loss_s/tr_loss_{index}.png')
    plt.close()

def load(model_save_path, optimizer_save_path) :
    model.load_state_dict(torch.load(model_save_path))
    optimizer.load_state_dict(torch.load(optimizer_save_path))

def validate(folds, test_loader) :
    cross_val = []
    all_labels = []
    all_predictions = []
    model.eval()

    for fold in range(folds) : 
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        cross_val.append(100 * correct / total)

    return cross_val, all_labels, all_predictions

#folds = 5
#cross_val, all_labels, all_predictions = validate(folds)
# print(f'Accuracy of the model on the test set (5 folds): {sum(cross_val)/len(cross_val):.2f}%')


def full_metrics(all_labels, all_predictions, index, plot = True) :
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted',zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted',zero_division=0)

    if plot :
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'{saving_folder}confusion_{index}.png')
    
    return precision, recall, f1, conf_matrix

def prediction(X_test) : 

    def predict(input_vector, label_encoder):
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_vector).float()
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            _, predicted_class = torch.max(output.data, 0)
            return label_encoder.inverse_transform([predicted_class.item()])[0]
    
    example_vector = X_test[0]
    predicted_class = predict(example_vector)
    print(f'Predicted class: {predicted_class}')
    return predicted_class

def full_metrics_bis(all_labels, all_predictions, index, label_encoder, plot=True):
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # Compute metrics per class
    class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    class_metrics = {
        'class': label_encoder.inverse_transform(range(len(class_precision))),
        'precision': class_precision,
        'recall': class_recall,
        'f1': class_f1
    }

    if plot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'{saving_folder}confusion_{index}.png')
    
    return precision, recall, f1, conf_matrix, class_metrics


data = []
limits = [10, 20, 30, 50, 65, 80, 100]
for l in limits : 
    for index in tqdm(range(3,4)): 

        print(f'\n Index = {index} // tax = {tax_classes[index]}')
        
        limit = l 
        X_train, X_test, y_train, y_test, train_loader, test_loader, num_classes, label_encoder = data_loading(index, taxonomy, tensor, limit)
        print(len(y_test))
        model = SimpleNN(input_size=1280, num_classes=num_classes, intermedita_size = 2560)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1200
        print(num_classes)
        model_save_path = saving_folder+'weights_s/'+f'classifier_nn_{index}.pth'
        optimizer_save_path = saving_folder+'weights_s/'+f'classifier_nn_optimizer_{index}.pth'
        folds = 5

        train(train_loader, num_epochs, model_save_path, optimizer_save_path, index)
        load(model_save_path, optimizer_save_path)
        cross_val, all_labels, all_predictions = validate(folds, test_loader)
        accuracy = sum(cross_val)/len(cross_val)
        precision, recall, f1, conf_matrix = full_metrics(all_labels, all_predictions, index, plot = False)
        print([index, tax_classes[index], accuracy, precision, recall, f1])
        data.append([index, tax_classes[index], accuracy, precision, recall, f1])

        precision, recall, f1, conf_matrix, class_metrics = full_metrics_bis(all_labels, all_predictions, index, label_encoder, plot=False)

        #print("\nMetrics per class:")
        #for cls, prec, rec, f1 in zip(class_metrics['class'], class_metrics['precision'], class_metrics['recall'], class_metrics['f1']):
        #    print(f"specie: {cls} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")


for p in data : 
    print(p)

csv_file = f"{saving_folder}training_results.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["taxonomy index", "taxonomy class", "accuracy", "precision", "recall", "f1 score"])  
    for row in data:
        writer.writerow(row)  