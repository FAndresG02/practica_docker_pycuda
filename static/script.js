
document.getElementById('inputMascara').addEventListener('keydown', function(event) {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') {
      event.preventDefault();
    }
});

// Seleccionamos el input que contiene la imagen
const inputFile = document.getElementById('imagen');
const uploadedImage = document.getElementById('uploadedImage');
const imageResult = document.getElementById('imageResult');

const tiempo = document.getElementById('tiempo');
const mascara = document.getElementById('mascara');
const bloques = document.getElementById('bloques');
const hilos = document.getElementById('hilos');


inputFile.addEventListener('change', function() {

  // Verificamos si se ha seleccionado la imagen
  const file = inputFile.files[0];

  const formData = new FormData();
  formData.append('imagen', file);

  fetch('/upload_image', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      //Vizualizamos la imagen subida
      uploadedImage.src = 'data:image/png;base64,' + data.image;
      uploadedImage.style.display = 'block';
  })
  .catch(error => {
      console.error('Error:', error);
  });

});


function procesarDatos() {
    const select = document.getElementById('items');
    const input = document.getElementById('inputMascara');

    const valorSelect = select.value;
    const valorInput = input.value;

    const datosProcesamiento = new FormData();
    datosProcesamiento.append('tipoFiltro', valorSelect);
    datosProcesamiento.append('mascara', valorInput);

    fetch('/procesar_imagen', {
        method: 'POST',
        body: datosProcesamiento
    })
    .then(response => response.json())
    .then(data => {
        imageResult.src = 'data:image/png;base64,' + data.image;
        imageResult.style.display = 'block';

        tiempo.innerHTML = '<strong>Tiempo de procesamiento:</strong> ' + parseFloat(data.tiempo).toFixed(2) + ' ms';
        mascara.innerHTML = '<strong>Máscara:</strong> ' + data.mascara + 'x' + data.mascara;
        bloques.innerHTML = '<strong>Bloques:</strong> ' + data.bloques;
        hilos.innerHTML = '<strong>Hilos:</strong> ' + data.hilos;
        
    })
    .catch(error => {
        console.error('Error:', error);
    });
    
}

document.addEventListener('DOMContentLoaded', function() {
    const boton = document.getElementById('procesarBtn');
    boton.addEventListener('click', procesarDatos);
});