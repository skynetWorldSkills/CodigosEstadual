import cv2
import numpy as np

# Função para identificar formas
def identificar_forma(contorno):
    perimetro = cv2.arcLength(contorno, True)
    aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    lados = len(aprox)
    
    if lados == 3:
        return "Triângulo"
    elif lados == 4:
        _, _, w, h = cv2.boundingRect(aprox)
        proporcao = w / float(h)
        return "Quadrado" if 0.9 <= proporcao <= 1.1 else "Retângulo"
    elif lados == 5:
        return "Pentágono"
    elif lados > 5:
        return "Círculo"
    return "Desconhecido"

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para escala de cinza e aplicar Canny
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(blur, 50, 150)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        if cv2.contourArea(contorno) > 1000:  # Filtrar pequenos ruídos
            forma = identificar_forma(contorno)
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
                cv2.putText(frame, forma, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar resultado
    cv2.imshow("Detecção de Formas", frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
