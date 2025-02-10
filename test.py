import cv2
import numpy as np
import matplotlib as mt

# Feature 1 - Cor principal - Feito
# Feature 2 - Cor secundária - Feito
# Feature 3 - Formato - Feito 
# Feature 4 - Tamanho - Feito
# Feature 5 - Se há furos - Não iniciado

def identificar_abacaxi(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    contorn = imagem
    
#===============================================FEATURE 1 E FEATURE 2===========================================================
    preto_inf = np.array([42, 255, 0])
    preto_sup = np.array([62, 255, 19])
    
    vermelho_inf = np.array([0, 171, 99])
    vermelho_sup = np.array([7, 255, 199])
    
    verde_inf = np.array([57, 113, 55])
    verde_sup = np.array([71, 255, 154])
    
    amarelo_inf = np.array([24, 159, 169])
    amarelo_sup = np.array([32, 226, 239])
    
    laranja_inf = np.array([13, 204, 166])
    laranja_sup = np.array([23, 255, 255])
    
    preto = cv2.inRange(hsv, preto_inf, preto_sup)
    amarelo = cv2.inRange(hsv, amarelo_inf, amarelo_sup)
    vermelho = cv2.inRange(hsv, vermelho_inf, vermelho_sup)
    verde = cv2.inRange(hsv, verde_inf, verde_sup)
    laranja = cv2.inRange(hsv, laranja_inf, laranja_sup)

    contagem_preta = cv2.countNonZero(preto)
    contagem_amarela = cv2.countNonZero(amarelo)
    contagem_vermelho = cv2.countNonZero(vermelho)
    contagem_verde = cv2.countNonZero(verde)
    contagem_laranja = cv2.countNonZero(laranja)

    if contagem_amarela > contagem_verde and contagem_amarela > contagem_vermelho and contagem_amarela > contagem_laranja:
        feature_1 = "amarelo"
    elif contagem_verde > contagem_amarela and contagem_verde > contagem_vermelho and contagem_verde > contagem_laranja:
        feature_1 = "verde"
    elif contagem_vermelho > contagem_amarela and contagem_vermelho > contagem_verde and contagem_vermelho > contagem_laranja:
        feature_1 = "vermelho"
    elif contagem_laranja > contagem_amarela and contagem_laranja > contagem_verde and contagem_laranja > contagem_vermelho:
        feature_1 = "laranja"
    else:
        feature_1 = "null"
    
    if contagem_preta > contagem_verde:
        feature_2 = "preto"
    elif contagem_verde > contagem_preta and contagem_verde < 50000:
        feature_2 = "verde"
    else:
        feature_2 = "null"
        
    print("VALOR AMARELO: ", contagem_amarela, "  VALOR PRETO: ", contagem_preta, " VALOR RED:", contagem_vermelho, " VALOR VERDE:", contagem_verde, " VALOR ORANGE:", contagem_laranja)
    
    
#==========================================================================================================

#=================================================FEATURE 3 E FEATURE 4=========================================================

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    if contornos:
        maxArea = cv2.contourArea(contornos[0])
        contourMaxAreaId = 0
    
    for i, cnt in enumerate(contornos):
        
        if maxArea < cv2.contourArea(cnt):
                maxArea = cv2.contourArea(cnt)
                contourMaxAreaId = i
        
        cntMaxArea = contornos[contourMaxAreaId]
        
        perimetro = cv2.arcLength(cntMaxArea, True)
        approx = cv2.approxPolyDP(cntMaxArea, 0.02 * perimetro, True)
        cv2.drawContours(contorn, [cntMaxArea], -1, (0, 255, 0), 3)
        
        qnt_contornos = len(approx)
        # print(qnt_contornos)
        
        if qnt_contornos <= 6:
            feature_3 = "quadratico"
        elif qnt_contornos <= 8 and qnt_contornos > 6:
            feature_3 = "oval"
        elif qnt_contornos <= 14 and qnt_contornos > 8:
            feature_3 = "circular"
        else:
            feature_3 = "null"
        
        x, y, w, h = cv2.boundingRect(cntMaxArea)
        area = w * h
        
        if area <=100:
            feature_4 = "pequeno demais"
        elif area > 100 and area <= 10000:
            feature_4 = "pequeno"
        elif area > 10000 and area <= 30000:
            feature_4 = "medio"
        elif area > 30000 and area <= 60000:
            feature_4 = "grande"
        else:
            feature_4 = "null"
    
    else:
        feature_3 = "null"
        feature_4 = "null"
        
#=================================================FEATURE 4=========================================================

#==========================================================================================================

#==============================================ESCOLHA DO OBJETO============================================================

    if feature_2 == "preto":
        coroa = "coroa preta"
    elif feature_2 == "verde":
        coroa = "coroa verde"
    else:
        coroa = "camera"
    
    match feature_1:
        case "amarelo":
            abacaxi = "Abacaxi amarelo com "

        case "vermelho":
            abacaxi = "Abacaxi vermelho com "
        
        case "laranja":
            abacaxi = "Abacaxi laranja com "
            
        case "verde":
            abacaxi = "Abacaxi verde com "
        
        case _:
            abacaxi = "Nada na "
    
    objeto = abacaxi + coroa
    
    if contagem_verde > 59999:
        objeto = "Abacaxi verde com coroa verde"
    elif contagem_verde < 59999 and contagem_verde > 50000:
        objeto = "Abacaxi verde com coroa preta"
        
    #print("  Ver Feature 1:", feature_1, "  Ver Feature 2:", feature_2, " Ver Feature 3:", feature_3, " Ver feature 4", feature_4)

    return objeto, gray, thresh, amarelo

#==========================================================================================================
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultado, gray, thresh, amarelo = identificar_abacaxi(frame)

    cv2.putText(frame, resultado, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (20, 56), (618, 460), (0, 0, 255), 2)
    
    cv2.imshow("Deteccao de frutas", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("am", amarelo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
