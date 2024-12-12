import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations
from collections import Counter
from sklearn.utils import resample

# Função para calcular distâncias entre todas as combinações de pontos (com 2D - x e y)
def calculate_distances(landmarks):
    distances = []
    for (i, j) in combinations(range(len(landmarks)//2), 2):
        x1, y1 = landmarks[2*i], landmarks[2*i + 1]
        x2, y2 = landmarks[2*j], landmarks[2*j + 1]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)
    return distances

# Carregando o dataset
data_dict = pickle.load(open('./data1.pickle', 'rb'))

# Convertendo os labels para inteiros, se necessário
data_dict['labels'] = list(map(int, data_dict['labels']))

# Verificando a distribuição de classes
class_counts = Counter(data_dict['labels'])
print("Distribuição de classes:", class_counts)

# Verificando o tamanho de cada amostra
data_lengths = [len(item) for item in data_dict['data']]
print("Tamanhos das amostras:", Counter(data_lengths))

# Usando amostras com 42 elementos (assumindo landmarks em 2D)
expected_length = 42

# Filtrando amostras que têm o tamanho esperado
consistent_data = [item for item in data_dict['data'] if len(item) == expected_length]
consistent_labels = [data_dict['labels'][i] for i in range(len(data_dict['data'])) if len(data_dict['data'][i]) == expected_length]

# Verificando quantas amostras são consistentes
print(f"Número de amostras consistentes: {len(consistent_data)}")

# Se não houver dados consistentes, interrompa o código
if not consistent_data:
    print("Erro: Não foram encontrados dados consistentes.")
    exit(1)

# Transformando landmarks em distâncias entre pares de pontos
data_with_distances = [calculate_distances(item) for item in consistent_data]

# Reequilibrando o dataset com Data Augmentation e oversampling
data_list = list(data_with_distances)
labels_list = list(consistent_labels)

class_counts = Counter(labels_list)
max_class_size = max(class_counts.values())

balanced_data = []
balanced_labels = []

for label in class_counts:
    class_data = [data_list[i] for i in range(len(labels_list)) if labels_list[i] == label]
    
    # Fazer oversampling para equilibrar
    class_data_resampled = resample(class_data, replace=True, n_samples=max_class_size, random_state=42)
    
    balanced_data.extend(class_data_resampled)
    balanced_labels.extend([label] * max_class_size)

balanced_data = np.asarray(balanced_data)
balanced_labels = np.asarray(balanced_labels)

# Dividindo o conjunto de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(balanced_data, balanced_labels, test_size=0.2, shuffle=True, stratify=balanced_labels)

# Treinando um RandomForest com os dados de distâncias
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)# test n_stimatos = 300, 
rf_model.fit(x_train, y_train)

# Fazendo predições no conjunto de teste
y_predict = rf_model.predict(x_test)

# Calculando a acurácia
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% de acurácia.')

# Gerando o relatório de classificação
print("\nRelatório de Classificação:\n", classification_report(y_test, y_predict))

# Exibindo a matriz de confusão
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_predict))

# Salvando o modelo treinado
with open('rf_model1.p', 'wb') as f:
    pickle.dump({'model': rf_model}, f)
