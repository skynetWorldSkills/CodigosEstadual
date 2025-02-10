import cv2
import numpy as np

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar threshold binário inverso
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos e hierarquia
    contornos, hierarquia = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Verificar se há contornos
    for i, contorno in enumerate(contornos):
        # Obter o bounding box
        x, y, w, h = cv2.boundingRect(contorno)

        # Se for um contorno EXTERNO
        if hierarquia[0][i][3] == -1:  # -1 indica que não tem pai (é externo)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Objeto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 0, 255), -1)  # Marcar buracos

    # Contar quantos buracos existem
    num_buracos = sum(1 for i in range(len(contornos)) if hierarquia[0][i][3] != -1)

    # Classificar se há furos
    if num_buracos > 0:
        cv2.putText(frame, "Com Furos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Sem Furos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar a imagem
    cv2.imshow("Deteccao de Furos", frame)
    cv2.imshow("Imagem", thresh)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
