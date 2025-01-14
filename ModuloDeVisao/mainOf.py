import numpy as np
import cv2

ESC_KEY = 27

cap = cv2.VideoCapture(0)

#O método computeTracking(frame, hue, sat, val) tem a função de retornar tanto a máscara que captura o intervalo de cor determinado
#tanto a imagem real com um retângulo desenhado para a visualização do que será captado com os valores HSV dos objetos desejados.
#O método em si tem um dicionário com os valores mínimos e máximo e o objeto do qual pertencem, na linha 38, o método passa por 
#todas as máscaras com os objetos do dicionário, localizando o que mais se encaixa com o que está aparecendo no frame e colocando
#o retângulo e o nome do objeto que foi detectado
#Neste método, a gravação frame a frame é colocada na váriavel local frame que recebe os tratamentos de imagem a seguir:
# 1 - Conversão da imagem em RBG para o padrão de cores HSV
# 2 - É criada uma máscara com os valores que estão sendo capturados pelas trackbars pela função setLimitsOfTrackbar() e estão
# sendo guardados nas variáveis hue, sat e val.
# 3 - A máscara passa por tratamentos de erosão, dilatação e transformações morfológicas para a exclusão de ruídos
# 4 - A máscara é aplicada em cima da imagem, e os pixels que estão em valor lógico 1 por causa das cores selecionadas são
# selecionados por causa da limiarização e com isso, usamos uma função para gerar os contornos, pegamos o contorno com maior area
# selecionando-o e criando um retângulo vermelho em cima dele
def computeTracking(frame):
    
    kernel = np.ones((5,5), np.uint8)
    
    color_ranges = {
    "Post it rosa":([168, 99, 193], [199, 163, 255]),
    "Post it verde":([38, 26, 167], [62, 78, 231]),
    "Maca amarela":([17, 80, 193], [35, 182, 232]),
    "Pente roxo":([109, 40, 147], [135, 111, 194]),
    "Multimetro azul":([95, 128, 94], [109, 215, 172]),
    "Antitranspirante verde-escuro":([73, 43, 28], [102, 109, 165]),
    "Capa da Raspberry":([0, 82, 106], [19, 127, 184])
}
    
    # Converte a imagem de RGB para HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in color_ranges.items():
        # Define os intervalos de cores que vão aparecer na imagem final
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        
        # Cria a máscara para os pixels dentro do intervalo
        mask = cv2.inRange(hsvImage, lower, upper)
        
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        
        # Aplica a máscara na imagem
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Aplica limiarização para gerar uma imagem binária
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Encontra contornos nas regiões brancas da imagem binária
        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            maxArea = cv2.contourArea(contours[0])
            contourMaxAreaId = 0
            
            # Para cada contorno, verifica o de maior área
            for i, cnt in enumerate(contours):
                if maxArea < cv2.contourArea(cnt):
                    maxArea = cv2.contourArea(cnt)
                    contourMaxAreaId = i
                    
            # Obtenha o contorno com a maior área
            cntMaxArea = contours[contourMaxAreaId]
            
            # Desenha um retângulo ao redor do maior contorno
            xRect, yRect, wRect, hRect = cv2.boundingRect(cntMaxArea)
            cv2.rectangle(frame, (xRect, yRect), (xRect + wRect, yRect + hRect), (0, 0, 0), 2)
            cv2.putText(frame, color_name, (xRect, yRect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame, gray

while True:
    success, frame = cap.read()

    # Aplica o processamento de rastreamento
    frame, gray = computeTracking(frame)

    # Exibe as imagem processada
    cv2.imshow("Webcam", frame)

    # Encerra o loop ao pressionar a tecla 'q' ou ESC
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ESC_KEY:
        break

cap.release()
cv2.destroyAllWindows()

