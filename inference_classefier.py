import cv2
import mediapipe as mp
import numpy as np
import pickle
from itertools import combinations

# Função para calcular distâncias 3D entre combinações de landmarks
def calculate_3d_distances(landmarks):
    distances = []
    for (i, j) in combinations(range(len(landmarks) // 3), 2):
        x1, y1, z1 = landmarks[3 * i], landmarks[3 * i + 1], landmarks[3 * i + 2]
        x2, y2, z2 = landmarks[3 * j], landmarks[3 * j + 1], landmarks[3 * j + 2]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        distances.append(distance)
    return distances

# Carregar o modelo treinado
with open('rf_model1.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Inicializando MediaPipe e captura de vídeo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Dicionário de rótulos
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar landmarks da mão
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extrai coordenadas dos landmarks
            combined_landmarks = []
            for landmark in hand_landmarks.landmark:
                combined_landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Verifica o número de landmarks
            if len(combined_landmarks) != 63:
                print(f"Erro: Número incorreto de landmarks detectados. Detectado: {len(combined_landmarks) // 3} landmarks.")
                continue

            # Gera as distâncias 3D entre os landmarks para classificação
            distances = calculate_3d_distances(combined_landmarks)
            if len(distances) != 210:
                print(f"Erro: O número de características geradas ({len(distances)}) não corresponde ao esperado (210).")
                continue

            # Convertendo as distâncias para numpy array e fazendo a predição
            data_aux = np.asarray(distances).astype(np.float32).reshape(1, -1)
            prediction = model.predict(data_aux)
            predicted_label = int(prediction[0])
            predicted_character = labels_dict.get(predicted_label, "Desconhecido")

            # Exibir o gesto reconhecido no canto superior esquerdo da tela
            text_position = (20, 40)
            cv2.putText(frame, predicted_character, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)  # Borda preta
            cv2.putText(frame, predicted_character, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)  # Texto branco

            # Opcional: Desenhar os landmarks e conexões da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Reconhecimento de Gestos', frame)

    if cv2.waitKey(10) & 0xFF == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
