import cv2
import numpy as np

# Inicializar a captura da webcam
cap = cv2.VideoCapture(1)  # Use 0 para a webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se a captura falhar
    
    # Redimensionar para processamento mais rápido
    frame = cv2.resize(frame, (640, 480))

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar CLAHE para melhorar o contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Aplicar suavização (reduz ruídos)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar Transformada Top-Hat para destacar os buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    # Aplicar binarização para segmentação
    thresh = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por tamanho e formato circular
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if 100 < area < 1000 and 0.6 < circularity < 1.2:
            filtered_contours.append(c)

    # Desenhar os buracos detectados na imagem original
    result = frame.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 2)

    # Mostrar os resultados
    cv2.imshow("gray", gray)
    cv2.imshow("blurred", blurred)
    cv2.imshow("thresh", thresh)
    
    cv2.imshow("Detecção de Buracos do Abacaxi", result)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
