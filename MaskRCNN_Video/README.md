## Preparación del entorno

Prepararemos un entorno con python 3.7.7, Tensorflow 2.1.0 y keras 2.3.1

    $ conda create -n MaskRCNN anaconda python=3.7.7
    $ conda activate MaskRCNN
    $ conda install ipykernel
    $ python -m ipykernel install --user --name MaskRCNN --display-name "MaskRCNN"
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
    
## Instalar MaskRCNN

    $ python setup.py install
    $ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

## Prueba del modelo entrenado con custom-dataset
        
-   PARA PRUEBA DEL SISTEMA EN VIDEO:

    Modificar los parámetros 
    
    -   model_filename = "mask_rcnn_object_0040.h5" # Aquí deben cargar el modelo entrenado con su dataset
    -   class_names = ['BG', 'persona'] # Las clases relacionadas con su modelo BG + clases custom
    -   min_confidence = 0.86 # Nivel mínimo de confianza para aceptar un hallazgo como positivo
    -   camera = cv2.VideoCapture("video.mp4") # Si desean correr un video cargandolo desde su PC
    
    $ python personaVideoDistancia.py
     
# Agradecimientos

    Matterport, Inc
    https://github.com/matterport

    Make Sense AI
    https://github.com/SkalskiP/make-sense

