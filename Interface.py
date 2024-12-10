import tkinter as tk
from tkinter import Label, Frame
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image, ImageTk
from itertools import combinations

# Carrega o modelo treinado
with open('rf_model1.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Mapeamento de classes para letras
class_to_letter = {i: chr(65 + i) for i in range(26)}

# Configurações do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Função para calcular distâncias 3D entre landmarks
def calculate_3d_distances(landmarks):
    distances = []
    for i, j in combinations(range(len(landmarks) // 3), 2):
        x1, y1, z1 = landmarks[3 * i], landmarks[3 * i + 1], landmarks[3 * i + 2]
        x2, y2, z2 = landmarks[3 * j], landmarks[3 * j + 1], landmarks[3 * j + 2]
        distances.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    return distances

# Classe da interface
class RealTimeInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Gestos em Libras")
        self.root.configure(bg="#2c3e50")
        
        # Estilo da fonte e cores
        self.font = ("Helvetica", 16, "bold")
        self.text_color = "#ecf0f1"
        self.bg_color = "#34495e"

        # Frame de vídeo
        self.video_frame = Frame(root, bg=self.bg_color)
        self.video_frame.pack(pady=10)

        # Label para o vídeo
        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        # Label para o resultado da predição
        self.result_label = Label(root, text="", font=self.font, fg="#e74c3c", bg=self.bg_color)
        self.result_label.pack(pady=10)

        # Adiciona instruções para o usuário
        self.instruction_label = Label(root, text="Mova a mão para que o modelo detecte o gesto!", font=("Helvetica", 12), fg=self.text_color, bg=self.bg_color)
        self.instruction_label.pack(pady=5)

        # Iniciar captura de vídeo
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Verifica se landmarks foram detectadas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extrai coordenadas e calcula distâncias
                    landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                    distances = calculate_3d_distances(landmarks)
                    
                    # Predição e atualização da interface
                    class_prediction = model.predict([distances])[0]
                    letter_prediction = class_to_letter.get(class_prediction, "Desconhecido")
                    self.result_label.config(text=f"Letra detectada: {letter_prediction}")

                    # Desenha as landmarks
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Exibe o vídeo na interface
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def __del__(self):
        self.cap.release()

# Inicializa e executa a interface
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeInferenceApp(root)
    root.mainloop()
