#Importação das bibliotecas
import numpy as np
import cv2

#Número da tecla Esc, para a interrupção do programa
ESC_KEY = 27

#Captura do vídeo da webcam
cap = cv2.VideoCapture(1)

#Imagens das frutas para testes em imagens
alho = cv2.imread('frutas/alho.png')
banana = cv2.imread('frutas/banana.png')
kiwi = cv2.imread('frutas/kiwi.png')
laranja = cv2.imread('frutas/laranja.png')
limao = cv2.imread('frutas/limao.png')
maca = cv2.imread('frutas/maca.png')
mangostim = cv2.imread('frutas/mangostim.png')
pessego = cv2.imread('frutas/pessego.png')
pimenta = cv2.imread('frutas/pimenta.png')
pomegranate = cv2.imread('frutas/pomegranate.png')

#O método setLimitsOfTrackbar() tem a função de retornar os valores Hue, Saturation e Value das cores no sistema de cores
#HSV, usado nas trackbars criadas. Além disso, ele ocasiona uma proteção para que o valor mínimo registrado nunca seja
#maior que o valor máximo, para evitar erros tanto na lógica, na execução deste programa ou no programa de detecção
def setLimitsOfTrackbar():
    hue = {}
    hue["min"] = cv2.getTrackbarPos("Min Hue", trackbarWindow)
    hue["max"] = cv2.getTrackbarPos("Max Hue", trackbarWindow)
    
    if hue["min"] > hue["max"]:
        cv2.setTrackbarPos("Max Hue", trackbarWindow, hue["min"])
        hue["max"] = hue["min"]
    
    sat = {}
    sat["min"] = cv2.getTrackbarPos("Min Saturation", trackbarWindow)
    sat["max"] = cv2.getTrackbarPos("Max Saturation", trackbarWindow)
    
    if sat["min"] > sat["max"]:
        cv2.setTrackbarPos("Max Saturation", trackbarWindow, sat["min"])
        sat["max"] = sat["min"]

    val = {}
    val["min"] = cv2.getTrackbarPos("Min Value", trackbarWindow)
    val["max"] = cv2.getTrackbarPos("Max Value", trackbarWindow)
    
    if val["min"] > val["max"]:
        cv2.setTrackbarPos("Max Value", trackbarWindow, val["min"])
        val["max"] = val["min"]
        
    return hue, sat, val

#O método computeTracking(frame, hue, sat, val) tem a função de retornar tanto a máscara que captura o intervalo de cor determinado
#tanto a imagem real com um retângulo desenhado para a visualização do que será captado com os valores destacados.
#Neste método, a gravação frame a frame é colocada na váriavel local frame que recebe os tratamentos de imagem a seguir:
# 1 - Conversão da imagem em RBG para o padrão de cores HSV
# 2 - É criada uma máscara com os valores que estão sendo capturados pelas trackbars pela função setLimitsOfTrackbar() e estão
# sendo guardados nas variáveis hue, sat e val.
# 3 - A máscara é aplicada em cima da imagem, e os pixels que estão em valor lógico 1 por causa das cores selecionadas são
# selecionados por causa da limiarização e com isso, usamos uma função para gerar os contornos, pegamos o contorno com maior area
# selecionando-o e criando um retângulo vermelho em cima dele
def computeTracking(frame, hue, sat, val):
    # Converte a imagem de RGB para HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define os intervalos de cores que vão aparecer na imagem final
    lowerColor = np.array([hue['min'], sat["min"], val["min"]])
    upperColor = np.array([hue['max'], sat["max"], val["max"]])
    
    # Cria a máscara para os pixels dentro do intervalo
    mask = cv2.inRange(hsvImage, lowerColor, upperColor)
    
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
        cv2.rectangle(frame, (xRect, yRect), (xRect + wRect, yRect + hRect), (0, 0, 255), 2)
    
    return frame, gray

# Criação da janela da trackbar, contendo as 6 trackbars para a alteração das máscaras
trackbarWindow = "Trackbar Window"
cv2.namedWindow(trackbarWindow)

def nothing(x):
    pass

# Cria os trackbars
cv2.createTrackbar("Min Hue", trackbarWindow, 0, 255, nothing)
cv2.createTrackbar("Max Hue", trackbarWindow, 255, 255, nothing)

cv2.createTrackbar("Min Saturation", trackbarWindow, 0, 255, nothing)
cv2.createTrackbar("Max Saturation", trackbarWindow, 255, 255, nothing)

cv2.createTrackbar("Min Value", trackbarWindow, 0, 255, nothing)
cv2.createTrackbar("Max Value", trackbarWindow, 255, 255, nothing)

while True:
    success, frame = cap.read()

    # Obtém os limites de cor das trackbars
    hue, sat, val = setLimitsOfTrackbar()

    # Aplica o processamento de rastreamento
    frame, gray = computeTracking(frame, hue, sat, val)

    # Exibe as imagens processadas
    cv2.imshow("Mask", gray)
    cv2.imshow("Webcam", frame)

    # Encerra o loop ao pressionar a tecla 'q' ou ESC
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ESC_KEY:
        break

cap.release()
cv2.destroyAllWindows()


# ==================== CÓDIGO FEITO PARA A OBTENÇÃO DE CORES EM IMAGENS ===========================

# import numpy as np
# import cv2

# ESC_KEY = 27

# # Carregar uma imagem para análise
# image_path = "frutas/limao.png"  # Altere para o caminho da sua imagem
# frame = cv2.imread(image_path)

# if frame is None:
#     print("Erro ao carregar a imagem.")
#     exit()

# # Nome da janela de trackbars
# trackbarWindow = "Trackbar Window"
# cv2.namedWindow(trackbarWindow)

# # Função de callback para os trackbars (não faz nada diretamente)
# def nothing(x):
#     pass

# # Criar os trackbars para ajustar os limites HSV
# cv2.createTrackbar("Min Hue", trackbarWindow, 0, 179, nothing)
# cv2.createTrackbar("Max Hue", trackbarWindow, 179, 179, nothing)

# cv2.createTrackbar("Min Saturation", trackbarWindow, 0, 255, nothing)
# cv2.createTrackbar("Max Saturation", trackbarWindow, 255, 255, nothing)

# cv2.createTrackbar("Min Value", trackbarWindow, 0, 255, nothing)
# cv2.createTrackbar("Max Value", trackbarWindow, 255, 255, nothing)

# # Função para definir os limites com base nos valores dos trackbars
# def setLimitsOfTrackbar():
#     hue = {}
#     hue["min"] = cv2.getTrackbarPos("Min Hue", trackbarWindow)
#     hue["max"] = cv2.getTrackbarPos("Max Hue", trackbarWindow)
    
#     if hue["min"] > hue["max"]:
#         cv2.setTrackbarPos("Max Hue", trackbarWindow, hue["min"])
#         hue["max"] = hue["min"]
    
#     sat = {}
#     sat["min"] = cv2.getTrackbarPos("Min Saturation", trackbarWindow)
#     sat["max"] = cv2.getTrackbarPos("Max Saturation", trackbarWindow)
    
#     if sat["min"] > sat["max"]:
#         cv2.setTrackbarPos("Max Saturation", trackbarWindow, sat["min"])
#         sat["max"] = sat["min"]

#     val = {}
#     val["min"] = cv2.getTrackbarPos("Min Value", trackbarWindow)
#     val["max"] = cv2.getTrackbarPos("Max Value", trackbarWindow)
    
#     if val["min"] > val["max"]:
#         cv2.setTrackbarPos("Max Value", trackbarWindow, val["min"])
#         val["max"] = val["min"]
        
#     return hue, sat, val

# # Função para processar a imagem e rastrear os objetos
# def computeTracking(frame, hue, sat, val):
#     # Converte a imagem de RGB para HSV
#     hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Define os intervalos de cores que vão aparecer na imagem final
#     lowerColor = np.array([hue['min'], sat["min"], val["min"]])
#     upperColor = np.array([hue['max'], sat["max"], val["max"]])
    
#     # Cria a máscara para os pixels dentro do intervalo
#     mask = cv2.inRange(hsvImage, lowerColor, upperColor)
    
#     # Aplica a máscara na imagem
#     result = cv2.bitwise_and(frame, frame, mask=mask)
    
#     # Aplica limiarização para gerar uma imagem binária
#     gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#     _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
#     # Encontra contornos nas regiões brancas da imagem binária
#     contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         maxArea = cv2.contourArea(contours[0])
#         contourMaxAreaId = 0
        
#         # Para cada contorno, verifica o de maior área
#         for i, cnt in enumerate(contours):
#             if maxArea < cv2.contourArea(cnt):
#                 maxArea = cv2.contourArea(cnt)
#                 contourMaxAreaId = i
                
#         # Obtenha o contorno com a maior área
#         cntMaxArea = contours[contourMaxAreaId]
        
#         # Desenha um retângulo ao redor do maior contorno
#         xRect, yRect, wRect, hRect = cv2.boundingRect(cntMaxArea)
#         cv2.rectangle(frame, (xRect, yRect), (xRect + wRect, yRect + hRect), (0, 0, 255), 2)
    
#     return frame, gray

# # Loop principal
# while True:
#     # Obtém os limites de cor das trackbars
#     hue, sat, val = setLimitsOfTrackbar()
    
#     # Aplica o processamento de rastreamento
#     processed_frame, gray = computeTracking(frame.copy(), hue, sat, val)
    
#     # Exibe as imagens processadas
#     cv2.imshow("Original", frame)
#     cv2.imshow("Máscara", gray)
#     cv2.imshow("Processado", processed_frame)
    
#     # Sai ao pressionar a tecla ESC
#     if cv2.waitKey(1) & 0xFF == ESC_KEY:
#         break

# # Fecha todas as janelas
# cv2.destroyAllWindows()
