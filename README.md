# object-detection
Cómo usar un modelo YOLO  pre-entrenado con OpenCV y detectar objetos en imágenes

Instalaciones necesarias:

    - Python
    - Numpy
    - OpenCV-Pyhon


Descargar los pesos pre-entrenados en la Terminal escribiendo:

`wget https://pjreddie.com/media/files/yolov3.weights`

Ejecutar el script:

`python yolo_opencv.py --image image1.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt`