# Importar paquetes necesarios
import cv2
import argparse
import numpy as np

# Manejar argumentos de línea de comando
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'ruta a la imagen de entrada')
ap.add_argument('-c', '--config', required=True,
                help = 'ruta al archivo de configuracion de yolo')
ap.add_argument('-w', '--weights', required=True,
                help = 'ruta a los presos pre-entrenados de yolo')
ap.add_argument('-cl', '--classes', required=True,
                help = 'ruta al archivo de texto que contiene nombres de clase')
args = ap.parse_args()

# Función para obtener los nombres de capa de salida en la arquitectura
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# Función para dibujar el cuadro delimitador en el objeto detectado con el nombre de la clase
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Leer la imagen de entrada
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# Leer nombres de clase de archivo de texto
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generar diferentes colores para diferentes clases
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Leer el modelo pre-entrenado y el archivo de configuración
net = cv2.dnn.readNet(args.weights, args.config)

# Crear blob (Binary Large Object) de entrada
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# Establecer blob de entrada para la red
net.setInput(blob)

# Ejecutar inferencia a través de la red y recopilar predicciones de las capas de salida
outs = net.forward(get_output_layers(net))

# Inicialización
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Para cada detección de cada capa de salida, obtener la presición, la identificación
# de la clase, los parámetros del cuadro delimitador e ignore las detecciones débiles (presición <0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Aplicar Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Ir a través de las detecciones restantes después de Non-Maximum Suppression
# y dibujar cuadro de delimitación
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# Mostrar imagen de salida
cv2.imshow("object detection", image)

# Esperar hasta que se pulse alguna tecla
cv2.waitKey()

# Guardar la imagen de salida
cv2.imwrite("object-detection.jpg", image)

# Liberar recursos
cv2.destroyAllWindows()