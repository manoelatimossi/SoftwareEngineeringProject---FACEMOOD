import cv2
import numpy as np


classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eyesClassifier = cv2.CascadeClassifier("haarcascade-eye.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostra = 25
id = input('type your ID') #person photos will be identified for an ID
largura, altura = 220, 220


 #Camera opening and face capturing
while(camera.isOpened()):
    conectado, imagem = camera.read()

    #imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BAYER_BG2BGR_EA) não sei se vai funcionar
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) #changes picture color to Black Grey Red

    facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.5, minSize=(100,100))

    for(x, y, l, a) in facesDetectadas:

        cv2.rectangle(imagem, (x, y), (x+l, y+a ),(0,0,255), 2) #face detection triangle
        region = imagem[y:y+a, x:x+l]
        greyregion = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        eyedetect = eyesClassifier.detectMultiScale(greyregion) #eye classifier
        for(ex,ey,el,ea) in eyedetect:
            cv2.rectangle(region,(ex, ey), (ex + el, ey + ea), (0, 255, 0), 2 )

            if cv2.waitKey(1) & 0xFF == ord(']'):
                if np.average(imagemCinza) > 110: #iluminação

                    FaceImage = cv2.resize(imagemCinza [y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa."+ str(id) + "." +str(amostra) + ".jpg", FaceImage)
                    print("[photo" + str(amostra)+ "sucessfully captured ]")
                    amostra += 1


    cv2.imshow("Face", imagem)

    cv2.waitKey(1)
    if (amostra >=numeroAmostra +1):
        break
camera.relase()

cv2.destroyAllWindows()
