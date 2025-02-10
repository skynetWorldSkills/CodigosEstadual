import cv2

# Lista para armazenar os pontos clicados
pontos = []

# Função de callback do mouse
def selecionar_ponto(event, x, y, flags, param):
    global pontos
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique esquerdo do mouse
        pontos.append((x, y))
        print(f"Ponto selecionado: {x}, {y}")
        
        if len(pontos) > 2:
            pontos = pontos[-2:]  # Mantém apenas os dois últimos pontos

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Cria a janela e define a função de callback do mouse
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", selecionar_ponto)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Desenha os pontos selecionados na imagem
    for p in pontos:
        cv2.circle(frame, p, 5, (0, 255, 0), -1)

    # Exibe a imagem com os pontos
    cv2.imshow("Webcam", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
