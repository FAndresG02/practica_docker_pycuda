from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from filtro_log import filtro_log_cuda
from filtro_gaussiano import filtro_gaussiano_cuda
from filtro_media import filtro_media_cuda

#drv.init()

app = Flask(__name__)

imageNumpy = None

imagen_final, bloques, hilos, duration = None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')


#Este método carga la imagen, luego la convierte a un arreglo de numpy y la guarda en la variable global imageNumpy
# y la retorna en formato base64 para poder mostrarla en el frontend
@app.route('/upload_image', methods=['POST'])
def upload_image():

    global imageNumpy
    
    file = request.files['imagen']
    
    # Leemos cargada la imagen
    image = Image.open(file.stream)
    
    # Convertimos la imagen a un arreglo de numpy
    imageNumpy = np.array(image)

    # Guardamos la imagen en memoria en formato PNG
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)


    # Codificamos la imagen en base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Retornamos la imagen codificada en base64 como parte de la respuesta JSON
    return jsonify({'image': img_base64}), 200


# Este método recibe el tipo de filtro y la mascara, llama al método correspondiente y retorna la imagen procesada 
# en formato base64, la mascara utilizada, el número de bloques, el número de hilos utilizados y el 
# tiempo de procesamiento
@app.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    
    global imageNumpy
    global imagen_final, bloques, hilos, duration
    
    tipoFiltro = request.form['tipoFiltro']
    mascara = request.form['mascara']
    mascara = int(mascara)

    imagenGrises = Image.fromarray(imageNumpy).convert("L")
    imagenGrises = np.array(imagenGrises)

    print("Tipo de filtro: ", tipoFiltro)

    if tipoFiltro == "filtroGaussiano":
        print("Filtro gaussiano")

        imagen_final, bloques, hilos, duration = filtro_gaussiano(mascara, imagenGrises)

    if tipoFiltro == "filtroMedia":
        print("Filtro media")
        
        imagen_final, bloques, hilos, duration = filtro_media(mascara, imagenGrises)

    if tipoFiltro == "filtroLog":
        print("Filtro LoG")

        imagen_final, bloques, hilos, duration = filtro_log(mascara, imagenGrises)
        

    return jsonify({
        'image': imagen_final,
        'mascara': mascara,
        'bloques': bloques,
        'hilos': hilos,
        'tiempo': duration
    }), 200

def filtro_log(mascara, imagenGrises):

    #Recuperamos los resultados del filtro log
    imagen_resultante, bloques, hilos, duration = filtro_log_cuda(mascara, imagenGrises)

    # Convertimos la imagen resultante a un formato que pueda ser enviado al frontend
    imagen_resultante = Image.fromarray(imagen_resultante)

    img_io = io.BytesIO()
    imagen_resultante.save(img_io, format='PNG')
    img_io.seek(0)

    # Codificamos la imagen en base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return img_base64, bloques, hilos, duration

def filtro_gaussiano(mascara, imagenGrises):

    #Recuperamos los resultados del filtro log
    imagen_resultante, bloques, hilos, duration = filtro_gaussiano_cuda(mascara, imagenGrises)

    # Convertimos la imagen resultante a un formato que pueda ser enviado al frontend
    imagen_resultante = Image.fromarray(imagen_resultante)

    img_io = io.BytesIO()
    imagen_resultante.save(img_io, format='PNG')
    img_io.seek(0)

    # Codificamos la imagen en base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return img_base64, bloques, hilos, duration

def filtro_media(mascara, imagenGrises):
    #Recuperamos los resultados del filtro media
    imagen_resultante, bloques, hilos, duration = filtro_media_cuda(mascara, imagenGrises)
    #Convertimos las imagen resultante a un formato que puede ser enviado al frontend
    imagen_resultante = Image.fromarray(imagen_resultante)

    img_io = io.BytesIO()
    imagen_resultante.save(img_io, format='PNG')
    img_io.seek(0)

    # Codificamos la imagen en base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    return img_base64, bloques, hilos, duration

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)

