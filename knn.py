#Import os para usar las funcionalidades de los path
import os
#import numpy para el manejo de array
import numpy as np

#Funcion obtenida del CIFAR-10
#File: Nombre del archivo a decifrar.
#Retorna: Un diccionario con data y labels
#Data: Una serie de 10000 arrays(imagenes) cada uno presenta una cantidad de 3072 colores.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Funcion Obtener Datos de la Imagen (array)
#Obtiene los datos para una imagen, ya sea su array de colores o su label
#-------Parametros-------
#dict nombre del batch a decifrar (de 1 a 5)
#img Si msg=b'data' --> numero de la imagen a obtener (de 0 a 9999), returna array de 3072 elemtos RGB
#img Si msg=b'labels' --> numero de la etiqueda de la imagen, para saber el nombre se puede usar getnomLbl enviando el resultado de esta funcion
#msg si es b'data' retorna los colores de la imagen img
#msg si es b'labels' retorna la etiqueta de la imagen img
#Retorna: Los colores o el nombre de la etiqueta, dependiendo del msg
def getData(dict, img, msg):
        diccionario = unpickle(os.getcwd()+'\\data_batch_'+str(dict))
        return diccionario[msg][img]
    
#Funcion similar a getData, pero obtiene los datos de prueba
#-------Parametros-------
#msg: Entre b'data' y b'labels' para el array de la imagen o el label
def getTest(msg):
    data = unpickle(os.getcwd()+'\\test_batch')
    return data[msg]

#Funcion obtener nombre de label
#Obtiene el nombre del label  partir del numero indicado
#-------Parametros-------
#num numero de la clase a buscar (del 0 al 9)
#retorna:  el nombre de la clase ej:truck, frog,...,cat
def getnomLbl(num):
        diccionario = unpickle(os.getcwd()+'\\batches.meta')
        return diccionario[b'label_names'][num]

#Funcion que crea un archivo y muestra la imagen
#-------Parametros-------
#getImg recibe el resutlado de la funcion getImg para mostrar la imagen indicada.
#Retorna: Nada (Muestra la imagen en pantalla)
def pintar(getImg):
    from PIL import Image
    data = np.zeros( (32,32,3), dtype=np.uint8)
    cont=0
    for i in range(0,32):
        for j in range(0,32):
            data[i][j] = [getImg[cont],getImg[cont+1024],getImg[cont+2048]]
            cont+=1
    img = Image.fromarray(np.asarray(data),'RGB')
    img.save('img.png')
    img.show()

#Clase K Nearest Neighbors
#Recibe un batch entero y lo inicializa en esta clase.
class KNN():
    def __init__(self):
        pass
    def entrenar(self, data, label):
        self.data = data
        self.label = label
    
#Funcion obtener datos de entrenamiento
#Crea un array con los datos de entrenamiento, donde un batch completo es
#una posicion en el arrat
#Retorna: Array con los datos.
def getTrainingSet():
    data = []
    for batch in range(1,6):
        dicc = unpickle(os.getcwd()+'\\data_batch_'+str(batch))
        k = KNN()
        k.entrenar(dicc[b'data'], dicc[b'labels'])
        data.append(k)
    return data


#Funcion Manhattan
#Obtiene la distancia/diferencia entre dos imagenes,
#obteniendo la suma entre los valores absolutos de las diferencias
#-------Parametros-------
#Es decir, entre sus pixeles
#img1: Imagen 1 a comparar
#img2: Imagen 2 a comparar con la 1
#retorna - entero: El valor total de la distancia de las dos imagenes
def manhattan(img1, img2):
    distancia = 0
    for i in range(3072):
        distancia += abs(img1[i]-img2[i])
    return distancia

#Funcion Chevyshev
#Obtiene la distancia/diferencia entre dos imagenes
#obteniendo el maximo entre los valores aboslutos de las diferencias.
#Es decir, entre sus pixeles
#-------Parametros-------
#img1: Imagen 1 a comparar
#img2: Imagen 2 a comparar con la 1
#retorna - entero: El valor total de la distancia de las dos imagenes
def chevyshev(img1, img2):
    max=0
    distancia = 0;
    for i in range(3072):
        distancia = abs(img1[i]-img2[i])
        if(distancia>max):
            max=distancia
    return max

#Funcion Levenshtein
#Obtiene la distancia/diferencia entre dos imagenes
#Es decir, entre sus pixeles
#-------Parametros-------
#img1: Imagen 1 a comparar
#img2: Imagen 2 a comparar con la 1
#retorna - entero: El valor total de la distancia de las dos imagenes
def levenshtein(img1,img2):
    d=dict()
    for i in range(len(img1)+1):
        d[i]=dict()
        d[i][0]=i
    for j in range(len(img2)+1):
        d[0][j]=j
    for i in range(1,len(img1)+1):
        for j in range(1,len(img2)+1):
            d[i][j] = min(d[i][j-1]+1,  d[i][j-1] + 1, d[i-1][j-1]+(not img1[i-1] == img2[j-1]))
    return d[len(img1)][len(img2)]

#-------Parametros-------
def getVecinos(img,tipo,k):
    import operator
    distancias = [] #donde se guardan la img con su distancia
    dist = 0
    datos = getTrainingSet()
    for i in range(1,6):
        for j in range(100): 
            imgN =datos[i-1].data[j]
            if(tipo==1):
                dist = manhattan(imgN,img)
            if(tipo==2):
                dist = chevyshev(imgN,img)
            if(tipo==3):
                dist = levenshtein(imgN,img)
            distancias.append((i-1,j,dist))
    #ordena todas las distancias en orden descendente
    distancias.sort(key=operator.itemgetter(2))
    votos = {}
    #Suma los puntos a cada foto de cual es su posible clase
    for i in range(k):
        res=(distancias[i][0],distancias[i][1])
        if res in votos:
            votos[res]+=1
        else:
            votos[res]=1
    ordenados = sorted(votos.items(), key=operator.itemgetter(1), reverse=True)
    #retorna par ordenado, donede la posicion 0 es el batch, y la pos 1 es la imagen
    return datos[ordenados[0][0][0]].label[ordenados[0][0][1]]
#Funcion de inicio
#Realiza el proceso incial
#-------Parametros-------    
def Inicio(tipo=1,k=1):
    test = getTest(b'data')
    lbl = getTest(b'labels')
    hit=0
    cant = 5
    for i in (range(15,20)):
        getLbl = getVecinos(test[i],tipo,k)
        print("obtenido:" +getnomLbl(getLbl).decode() +" real: "+getnomLbl(lbl[i]).decode() )
        if(getLbl==lbl[i]):
            hit+=1
    print("hits: ",hit)
    res = (hit/float(cant))*100.0
    print("res: ",res)


Inicio(3,1)
#levenshtein(getTest(b'data')[15],getTest(b'data')[15])

