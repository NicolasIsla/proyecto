# proyecto
Proyecto 15. Representación temporal de alertas ZTF

ALeRCE es un broker de alertas astronómicas que recibe y procesa observaciones provenientes del survey ZTF en tiempo real. Dentro de los datos entregados por las alertas se encuentran los stamps o imágenes del lugar donde se produce la alerta. Estos stamps están compuestos por 3 imágenes: La imagen de referencia, la cual corresponde a un promedio de la posición antes de la alerta, imagen de ciencia correspondiente a una foto en el momento en que se detecta la alerta y la imagen de diferencia, siendo una diferencia entre las dos anteriores. Un objeto puede ser detectado más de una vez, lo cual agrega un carácter temporal a las alertas. El objetivo del proyecto es utilizar un AutoEncoder compuesto por una parte convolucional que proceso las imágenes y una parte recurrente que pueda procesar las características recurrentes entregadas por ZTF. Como mínimo se espera que implementen un AutoEncoder convolucional de la primera alerta basándose en el modelo entregado en referencias [1] y exploren la representación generada en el espacio latente. La base de datos requiere ser preprocesada antes de poder ser utilizada.

Integrantes: Rodrigo Ortiz y Nicolas Isla

Para ejecutar el codigo se tiene que tener acceso al Drive del proyecto, de esta forma se puede utilizar Google Colab.
Se añade el link del https://drive.google.com/drive/folders/16_MnWlUUVc04LfG5tB_Gf_22PLsx5RkE?usp=sharing
