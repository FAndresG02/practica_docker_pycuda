import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from time import time

cuda.init()

def generate_gaussian_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    # Factor de normalización (1 / (2πσ²))
    normalization_factor = 1 / (2 * np.pi * sigma**2)

    total = 0.0
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            exponent = -(dx**2 + dy**2) / (2 * sigma**2)
            value = normalization_factor * np.exp(exponent)
            kernel[y, x] = value
            total += value

    # Normalizamos para que la suma del kernel sea 1
    kernel /= total
    return kernel

def filtro_gaussiano_cuda(mascara, image):

    device = cuda.Device(0)
    context = device.make_context()

    try:
        sigma = 0.3 * ((mascara - 1) * 0.5 - 1) + 0.8
        kernel = generate_gaussian_kernel(mascara, sigma)

        height, width = image.shape
        kCenterX = mascara // 2
        kCenterY = mascara // 2

        mod = SourceModule("""
        __global__ void convolutionKernel(
            const unsigned char* d_input,
            unsigned char* d_output,
            int width, int height,
            const float* d_kernel,
            int kWidth, int kHeight,
            int kCenterX, int kCenterY) {

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                float sum = 0.0f;
                for (int ky = 0; ky < kHeight; ky++) {
                    for (int kx = 0; kx < kWidth; kx++) {
                        int posX = x + kx - kCenterX;
                        int posY = y + ky - kCenterY;

                        if (posX >= 0 && posX < width && posY >= 0 && posY < height) {
                            float pixel = (float)(d_input[posY * width + posX]);
                            float kval = d_kernel[ky * kWidth + kx];
                            sum += pixel * kval;
                        }
                    }
                }

                int valor = (int)(roundf(sum));
                valor = min(max(valor, 0), 255);
                d_output[y * width + x] = (unsigned char)(valor);
            }
        }
        """)

        start = time()  # Medición de tiempo de inicio

        input_flat = image.astype(np.uint8).flatten()
        output_flat = np.empty_like(input_flat)
        kernel_flat = kernel.astype(np.float32).flatten()

        d_input = cuda.mem_alloc(input_flat.nbytes)
        d_output = cuda.mem_alloc(output_flat.nbytes)
        d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

        cuda.memcpy_htod(d_input, input_flat)
        cuda.memcpy_htod(d_kernel, kernel_flat)

        block_size = (16, 16, 1)
        grid_size = ((width + 15) // 16, (height + 15) // 16)

        kernel_func = mod.get_function("convolutionKernel")
        kernel_func(d_input, d_output,
                    np.int32(width), np.int32(height),
                    d_kernel,
                    np.int32(mascara), np.int32(mascara),
                    np.int32(kCenterX), np.int32(kCenterY),
                    block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(output_flat, d_output)

        end = time()  # Medición de tiempo de finalización
        duration_ms = (end - start) * 1000  # Convertir a milisegundos

        result_image = output_flat.reshape((height, width))

        # Imprimir el tiempo de ejecución en milisegundos
        print(f"Tiempo de ejecución: {duration_ms:.2f} ms")

        return result_image, grid_size, block_size, duration_ms

    finally:
        context.pop()
