import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def puntoMedio(point1, point2):
    neoX=int((point1[0]+point2[0])/2)
    neoY=int((point1[1]+point2[1])/2)
    return (neoX, neoY)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2+(point1[1] - point2[1])**2)

def RGB_a_EscalaGrises(img):
    Gris = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R+G+B)
    Gris = img.copy()

    for i in range(3):
        Gris[:, :, i] = Avg

    return Gris

def Binarizacion(image, umbral):
    Binaria = np.zeros(image.shape)
    Binaria = image.copy()
    Temp = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] < umbral:
            # if image[i][j] < umbral:
                Temp[i][j] = 0
            else:
                Temp[i][j] =255
    # for i in range(3):
        # Binaria[:, :, i] = Temp
    return Temp

image = cv2.imread("Jit1.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)

for i in range(len(centers)):
    if(centers[i][0]!=142):
        centers[i]=np.zeros(3,dtype=np.uint8)

labels = labels.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

bin = RGB_a_EscalaGrises(segmented_image)

blur = cv2.GaussianBlur(bin,(15,15),cv2.BORDER_DEFAULT,2)
kernel = np.ones((15,15), np.uint8)
opening = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
edged = cv2.Canny(closing, 30, 200)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

anterior=0
colors=[(0,0,255),(0,125,255),(0,255,0),(127,0,255)]
tomatos=[]
i=0
finder=image.copy()
for c in contours:
    if cv2.contourArea(c) < 10000 or abs(anterior-cv2.contourArea(c)) < 500:
        continue
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    tomatos.append(box)
    cv2.drawContours(finder,[box],0,colors[i],8)
    anterior=cv2.contourArea(c)
    i+=1

i=0
for caja in tomatos:
    tempstr="Jitomate "+str(i+1)
    if(i==0):
        print(tempstr)
        tempstr+=", D="+str(distance(puntoMedio(caja[0],caja[3]), puntoMedio(caja[1],caja[2])))
        image = cv2.line(image, puntoMedio(caja[0],caja[3]), puntoMedio(caja[1],caja[2]), colors[i], 8)
        image = cv2.putText(image, tempstr, puntoMedio(puntoMedio(caja[0],caja[3]),puntoMedio(caja[1],caja[2]+100)), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 8, cv2.LINE_AA)
        print("PuntoA"+str(puntoMedio(caja[0],caja[3])))
        print("PuntoB"+str(puntoMedio(caja[1],caja[2])))
        print("Distancia es de "+str(distance(puntoMedio(caja[0],caja[3]), puntoMedio(caja[1],caja[2]))))
    elif(i==2):
        print(tempstr)
        tempstr+=", D="+str(distance(puntoMedio(caja[0],caja[1]), puntoMedio(caja[2],caja[3])))
        image = cv2.line(image, puntoMedio(caja[0],caja[1]), puntoMedio(caja[2],caja[3]), colors[i], 8)
        image = cv2.putText(image, tempstr, puntoMedio(puntoMedio(caja[0],caja[1]),puntoMedio(caja[2],caja[3]+250)), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 8, cv2.LINE_AA)
        print("PuntoA"+str(puntoMedio(caja[0],caja[1])))
        print("PuntoB"+str(puntoMedio(caja[2],caja[3])))
        print("Distancia es de "+str(distance(puntoMedio(caja[0],caja[1]), puntoMedio(caja[2],caja[3]))))
    i+=1

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(3, 2, 1)
plt.imshow(segmented_image)
plt.axis('off')
plt.title("Segmentacion Kmeans")

fig.add_subplot(3, 2, 2)
plt.imshow(bin)
plt.axis('off')
plt.title("Binaria")

fig.add_subplot(3, 2, 3)
plt.imshow(blur)
plt.axis('off')
plt.title("Gauss")

fig.add_subplot(3, 2, 4)
plt.imshow(closing)
plt.axis('off')
plt.title("Morfologia Cerrada")

fig.add_subplot(3, 2, 5)
plt.imshow(edged)
plt.axis('off')
plt.title("Contornos")

fig.add_subplot(3, 2, 6)
plt.imshow(finder)
plt.axis('off')
plt.title("Rectangles")

plt.show()

plt.figure(2)
plt.imshow(image)
plt.axis('off')
plt.title("Final")
plt.show()
