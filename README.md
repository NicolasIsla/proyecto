# Proyecto 15. Representación temporal de alertas ZTF

Integrantes: Rodrigo Ortiz y Nicolas Isla

ALeRCE es un broker de alertas astronómicas que recibe y procesa observaciones provenientes del survey ZTF en tiempo real. Dentro de los datos entregados por las alertas se encuentran los stamps o imágenes del lugar donde se produce la alerta. Estos stamps están compuestos por 3 imágenes: La imagen de referencia, la cual corresponde a un promedio de la posición antes de la alerta, imagen de ciencia correspondiente a una foto en el momento en que se detecta la alerta y la imagen de diferencia, siendo una diferencia entre las dos anteriores. Un objeto puede ser detectado más de una vez, lo cual agrega un carácter temporal a las alertas. El objetivo del proyecto es utilizar un AutoEncoder compuesto por una parte convolucional que proceso las imágenes y una parte recurrente que pueda procesar las características recurrentes entregadas por ZTF.

El proyecto se encuentra dividido en 2 áreas, la primera la creación de un AutoEncoder con una rama de clasificación, y la segunda la creación de una red recurrente que procese las características temporales de las alertas, ambas partes se encuentran separadas en sus respectivas carpetas.

## Instrucciones de ejecución

Dentro de cada sección se encuentran 2 Jupyter Notebooks, los cuales consisten en uno de exploración y obtención de resultados, mientras que el otro consiste en el entrenamiento, el cual se hizo mediante Google Colab, debido a que se puede acceder al uso de una GPU. Finalmente, tambien se añaden los distintos modelos entrenados, donde se realizaron experimentos, para obtener el mejor modelo.

## Acceso a las bases de datos

### Bases de datos originales 

Link de acceso: https://drive.google.com/drive/folders/1vqfoxF-KyMNnLxABZZ_kb76DrUrixEud?usp=sharing}

Se puede encontrar los archivos td_ztf_stamp_17_06_20.pkl y recurrent_dataset.pk, los cuales contienen el muestreo de datos para llevar a cabo el proyecto. En el primer archivo contiene 3 sets de datos, entrenamiento, validación y prueba, donde hay cerca de 70000 imagenes de Science, Template y Difference. En el segundo archivo contiene tambien 70000 muestras, pero con una serie temporal añadida a cada una.

### Bases de datos codificadas 

Link de acceso: https://drive.google.com/drive/folders/1t0XthUcBawJ7nuFpCC0eTG1rV2r-2Oat?usp=sharing

Se puede encontrar los archivos base.npy y baseRecurrente.npy, que consiste en los dataset originales, pero pasados por el AutoEncoder entrenado.

