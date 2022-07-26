# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import argparse
from scipy.spatial import distance as dist

import colorsys
import cv2

from mrcnn.config import Config
from mrcnn import model as personModel

HEIGHT=840
WIDTH=680

MIN_DISTANCE=50

MIN_CONF = 0.8
NMS_THRESH = 0.8

centroides = list()
confidences = []

# construct the argument parse and parse the arguments
# construir el argumento parse y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="",	help="path to (optional) output video file")
args = vars(ap.parse_args(["--output","outputVideo.avi"]))

# MODELOS
#model_filename = "mask_rcnn_object_0005.h5"
#model_filename = "mask_rcnn_object_0005_200img.h5"
model_filename = "mask_rcnn_object_0010.h5"
#model_filename = "mask_rcnn_object_0030.h5"
#model_filename = "mask_rcnn_object_0040.h5"

class_names = ['BG', 'persona']
min_confidence = 0.86

#MODELO WIN
#camera = cv2.VideoCapture("video_bajaDensidad.mp4")
#camera = cv2.VideoCapture("video_3-4_metros_densidadMedia.mp4")
#camera = cv2.VideoCapture("video_3-4_metros_densidadMediaAlta.mp4")
#camera = cv2.VideoCapture("video_3-4_metros_Trim.mp4")
#camera = cv2.VideoCapture("video_densidadMedia.mp4")
camera = cv2.VideoCapture("video_filipinas1_densidadBaja.mp4")
#camera = cv2.VideoCapture("video_filipinas2_densidadMediaBaja.mp4")

#F EL MODELO
#camera = cv2.VideoCapture("video_tokyo1.mp4")
#camera = cv2.VideoCapture("video_timeSquare1_AltaDensidad.mp4")
#camera = cv2.VideoCapture("video_timeSquare2_bajaDensidad.mp4")
#camera = cv2.VideoCapture("video_timeSquare3_mediaDensidad.mp4")

class PersonaConfig(Config):
    # Give the configuration a recognizable name
    # Dar un nombre a la configuracion 
    NAME = "object"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    # Entrenar en 1 GPU y 1 imagen por GPU, el tamaño del Batch es 1 (GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # Numero de clases (inluyendo el background)
    NUM_CLASSES = 1 + 1  # background + 1 (persona)

    # All of our training images are 512x512
    # Tamaño de todas las imagenes son: 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    # Se puede modificar este valor para ver si el entrenamiento es mejorado
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space on saved models (in the MODEL_DIR), try making this value larger.
    # Esto se refiere al proceso de validacion. Si se esta usando mucha mas memoria en disco salvando modelos (exactamente MODEL_DIR), este valor se deberia incrementar.
    VALIDATION_STEPS = 5

    # Other esstential config
    # Otras configuraciones esenciales
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = PersonaConfig()
#config.display()

class InferenceConfig(PersonaConfig):
    # config para optimizar el conteo de personas
    BACKBONE_STRIDES=[4, 8, 16, 32, 64]
    BATCH_SIZE=1
    COMPUTE_BACKBONE_SHAPE=None
    DETECTION_MAX_INSTANCES=100
    DETECTION_MIN_CONFIDENCE=min_confidence
    DETECTION_NMS_THRESHOLD=0.3
    FPN_CLASSIF_FC_LAYERS_SIZE=1024
    GPU_COUNT=1
    GRADIENT_CLIP_NORM=5.0
    IMAGES_PER_GPU=1
    IMAGE_CHANNEL_COUNT=3
    IMAGE_MAX_DIM=512
    IMAGE_META_SIZE=14
    IMAGE_MIN_DIM=512
    IMAGE_MIN_SCALE=0
    LEARNING_MOMENTUM=0.9
    LEARNING_RATE=0.001
    MASK_POOL_SIZE=14
    MASK_SHAPE=[28, 28]
    MAX_GT_INSTANCES=100
    MINI_MASK_SHAPE=(56, 56)
    NAME="object"
    NUM_CLASSES=2
    POOL_SIZE=7
    POST_NMS_ROIS_INFERENCE=1000
    POST_NMS_ROIS_TRAINING=2000
    PRE_NMS_LIMIT=6000
    ROI_POSITIVE_RATIO=0.33
    RPN_ANCHOR_RATIOS=[0.5, 1, 2]
    RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE=1
    RPN_NMS_THRESHOLD=0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE=256
    STEPS_PER_EPOCH=500
    TOP_DOWN_PYRAMID_SIZE=256
    TRAIN_ROIS_PER_IMAGE=200
    VALIDATION_STEPS=5
    WEIGHT_DECAY=0.0001
    
inference_config = InferenceConfig()
inference_config.display()

# Recreate the model in inference mode
# Recreacion del modelo en modo inferencia
model = personModel.MaskRCNN(mode="inference", config=inference_config,  model_dir='logs')

# Get path to saved weights. Either set a specific path or find last trained weights
# Ruta hacia los pesos guardados. De igual manera la ruta hacia el ultimo modelo entrenado
model_path = os.path.join('logs', model_filename)
#model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
# Cargar los pesos entrenados (llenar la ruta de pesos establecidos aqui)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

def calcular_dist(boxes, indexes):

  if len(indexes) > 2:
    idf = indexes.flatten()
    centers = list()
    status = list()
    violate = set()

    safe = list()
    low_risk = list()
    high_risk = list()

    for i in idf:
    #The mask RCNN bounding box format demands the top left and bottom right coordinate of the box which is given by: [x, y, x+w, y+h].
    #El formato de la bbox de mask RCNN arroja el borde superior izquierda, y el borde inferior derecho en base a las coordenada dadas por: [x, y, x+w, y+h]
    # x, y, x+w, y+h
      (x,y) = (boxes[i][0],boxes[i][1]) # top-left position
      (w,h) = (boxes[i][2]-boxes[i][0],boxes[i][3]-boxes[i][1])
      
      centers.append([int(y+(h/2)),int(x+(w/2))])
      #print("centros ", centers)
      
    
    dst = dist.cdist(centers, centers, metric="euclidean")
    #print("matriz distancia \n", dst)
    
    # loop over the upper triangular of the distance matrix
    # Recorrer la matriz de distancia contando unicamente su traingularidad superior
    for i in range(0, dst.shape[0]):
        for j in range(i + 1, dst.shape[1]):
            # check to see if the distance between any two centroid pairs is less than the configured number of pixels
            # revisar si la distancia que existe entre dos distintos centroides es menor que la cantidad de pixeles configurados inicialmente.
            if dst[i, j] < MIN_DISTANCE:
                # update our violation set with the indexes of the centroid pairs
                # actualizamos el conjunto de "violate" con los indices de los centroides que cumplen esta condicion
                violate.add(i)
                violate.add(j)
                high_risk.append([centers[i], centers[j]])
                #y,  x, y+h x+w
                y1, x1, y2, x2 = boxes[i]
                #print("coordenadas ", boxes[i])
                cv2.rectangle(frame_obj, (x1, y1), (x2, y2),(0,10,255), 2)

    person_count = len(centers)
    safe_count = len(centers)-len(violate)
    high_risk_count = len(violate)
    
    return high_risk_count, safe_count, idf, high_risk

def draw_distance(frame, idf, boxes, WIDTH, HEIGHT, high_risk, safe, high_risk_cord, total):

  for i in idf:
    sub_img = frame[630:HEIGHT, 0:150]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.70, black_rect, 0.30, 1.0)
    frame[630:HEIGHT, 0:150] = res
    #                                                                        B- G- R-
    #cv2.putText(frame_obj,'TEST',(0, 670), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.putText(frame_obj, "TOTAL      : {}".format(total),(10, 650),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame_obj, "EN RIESGO  : {}".format(high_risk),(10, 670),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 10, 255), 2)
 
  for l in high_risk_cord:
    cv2.line(frame_obj, tuple(l[0]), tuple(l[1]), (10, 44, 236), 2)   

  return frame

writer = None

while camera:
    ret, frame = camera.read()
    frame = cv2.resize(frame, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
    
    results = model.detect([frame], verbose=0)
    r = results[0]
    
    N =  r['rois'].shape[0]
    #print("N:",N)
    boxes=r['rois']
    masks=r['masks']
    class_ids=r['class_ids']
    scores=r['scores']
    # Bounding box indexes. 
    # Indices de los cuadros delimitadores.
    indices = cv2.dnn.NMSBoxes(boxes, scores, MIN_CONF, NMS_THRESH)  
    #print("indices: ", indices)    
       
    hsv = [(i / N, 1, 0.7) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    
    random.shuffle(colors)
    #print("N_obj:",N)
    masked_image = frame.astype(np.uint32).copy()
    
    for i in range(N):
        
        #if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
        #    continue

        # lista de colores para cada mascara de las instancias por cada frame
        color = list(np.random.random(size=3) * 256)
        mask = masks[:, :, i]
        alpha=0.5
        
        # Recorrer las mascaras y darles una tonalidad que se refleje en los resultados.
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1, masked_image[:, :, c] * (1 - alpha) + alpha * color[c], masked_image[:, :, c])
          
        frame_obj=masked_image.astype(np.uint8)
        
        # Localizar los atributos de las detecciones
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        masked_image = frame_obj.astype(np.uint32).copy()
        
        # Incluir cada score al conjunto de confianza 
        confidences.append(float(score))
    
    # se calcula las distancias de todas las instancias en el frame dado
    high_risk, safe, idf, high_risk_cord = calcular_dist(boxes, indices)
    # se dibuja dicha distancia para que sea acorde 
    draw_distance(frame_obj, idf, boxes, WIDTH, HEIGHT, high_risk, safe, high_risk_cord, N)
    
    if N>0:
        cv2.imshow('Aglomeracion de personas', frame_obj)
    else:
        cv2.imshow('Aglomeracion de personas', frame)
    
    # parar la ejecucion del programa con letra 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break;
    
    # if an output video file path has been supplied and the video writer has not been initialized, do so now
    # si se proporcionó una ruta de archivo de video de salida y el escritor de video no se ha inicializado, se procede a configurarlo 
    if args["output"] != "" and writer is None:
        # initialize our video writer
        # inicializar el escritor de video 
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
    # si el escritor de video no es None, escriba el cuadro en el archivo de video de salida
    if writer is not None:
        writer.write(frame_obj)

# 
camera.release()
# destruir todas las ventanas y procesos despues de la inferencia.
cv2.destroyAllWindows()