import time
import cv2
import os

# Criar pasta para armazenar imagens
output_folder = "dataset\images\livro"
os.makedirs(output_folder, exist_ok=True)

# Iniciar webcam
cap = cv2.VideoCapture(0)
count = 0

while count < 100:  # Definir quantas imagens deseja capturar
    ret, frame = cap.read()
    if not ret:
        break

    img_name = os.path.join(output_folder, f"imagem_{count}.jpg")
    cv2.imwrite(img_name, frame)
    count += 1

    cv2.imshow("Captura de Imagens", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):  # Pressione 'q' para sair
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captura finalizada. {count} imagens salvas em {output_folder}.")