import cv2
import numpy as np

def detectar_contornos_ao_vivo():
    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(1)
    kernel = np.ones((5,5), np.uint8)  
      
    while True:
        # Capturar frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter para escala de cinza

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lowerColor = np.array([0, 0, 89])
        upperColor = np.array([183, 117, 206])
        
        # Cria a máscara para os pixels dentro do intervalo
        mask = cv2.inRange(hsvImage, lowerColor, upperColor)
        
        # Aplica a máscara na imagem
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Aplica limiarização para gerar uma imagem binária
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.erode(gray, kernel, iterations=2)
        gray = cv2.dilate(gray, kernel, iterations=2)
        
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # Desenhar contornos na imagem original
        cv2.drawContours(frame, contornos, -1, (0, 255, 0), 2)
        
        # Mostrar a imagem com contornos
        cv2.imshow('Contornos ao Vivo', frame)
        cv2.imshow("Thresh", mask)
        
        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Iniciar detecção ao vivo
detectar_contornos_ao_vivo()
