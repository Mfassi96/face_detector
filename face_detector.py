import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#XML con los datos para detectar la cara

img=cv2.imread('descarga.jpg')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# convierte la imagen en escalas de grises para mejorar la deteccion

faces=face_cascade.detectMultiScale(gray_img,
                                    scaleFactor=1.1,
                                    minNeighbors=5)

for x,y,w,h in faces:
    img=cv2.rectangle(
        img,
        (x,y),# coordenadas
        (x+w,y+h),#esquinas
        (0,0,255)# color del rectangulo BGR
        ,3#tamaño
        )

resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow("Foto",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()