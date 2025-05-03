import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from time import time

cuda.init() 


def generate_LoG_kernel(size, sigma):
    center = size // 2
    sigma2 = sigma ** 2
    sigma4 = sigma2 ** 2
    kernel = np.zeros((size, size), dtype=np.float32)

    total = 0.0
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            r2 = dx * dx + dy * dy
            value = (-1.0 / (np.pi * sigma4)) * (1 - (r2 / (2.0 * sigma2))) * np.exp(-r2 / (2.0 * sigma2))
            kernel[y, x] = value
            total += value

    mean = total / (size * size)
    kernel -= mean
    return kernel


def filtro_log_cuda(mascara, image):

    device = cuda.Device(0)
    context = device.make_context() 

    try:
        # Kernel CUDA
        mod = SourceModule("""
        __global__ void convolutionKernel(
            const unsigned char* d_input,
            unsigned char* d_output,
            int width, int height,
            const float* d_kernel,
            int kWidth, int kHeight,
            int kCenterX, int kCenterY,
            int startY, int endY) {

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                int procStartY = (startY > kCenterY ? startY : kCenterY);
                int procEndY = (endY < (height - kCenterY) ? endY : (height - kCenterY));

                if (y >= procStartY && y < procEndY) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < kHeight; ky++) {
                        for (int kx = 0; kx < kWidth; kx++) {
                            int posX = x + kx - kCenterX;
                            int posY = y + ky - kCenterY;
                            float pixel = (float)(d_input[posY * width + posX]);
                            float kval = d_kernel[ky * kWidth + kx];
                            sum += pixel * kval;
                        }
                    }

                    int valor = (int)(roundf(sum * 10.0f));
                    valor = min(max(valor, 0), 255);
                    d_output[y * width + x] = (unsigned char)(valor);
                }
            }
        }
        """)

        convolutionKernel = mod.get_function("convolutionKernel")
        height, width = image.shape

        # Generamos el kernel
        kernel_sigma = round((mascara / 6.0) * 100) / 100.0
        k_center = mascara // 2

        # Generamos el kernel
        kernel = generate_LoG_kernel(mascara, kernel_sigma).astype(np.float32)
        kernel_flat = kernel.ravel()

        # Reservamos memoria y copiamos a GPU
        d_input = cuda.mem_alloc(image.nbytes)
        d_output = cuda.mem_alloc(image.nbytes)
        d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

        cuda.memcpy_htod(d_input, image)
        cuda.memcpy_htod(d_kernel, kernel_flat)

        # Definimos el grid y el bloque
        block = (16, 16, 1)
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1], 1)

        startY = 0
        endY = height

        # Ejecutamos el kernel y medimos el tiempo
        start = time()

        convolutionKernel(
            d_input, d_output,
            np.int32(width), np.int32(height),
            d_kernel, np.int32(mascara), np.int32(mascara),
            np.int32(k_center), np.int32(k_center),
            np.int32(startY), np.int32(endY),
            block=block, grid=grid
        )

        cuda.Context.synchronize()
        duration = time() - start

        # Copiamos el resultado a CPU
        output_image = np.empty_like(image)
        cuda.memcpy_dtoh(output_image, d_output)


        print(f"Tiempo de ejecución: {duration * 1000:.2f} ms\n")
        
        bloques = 16
        hilos = 16
        duration = duration * 1000
        
        return output_image, bloques, hilos, duration

    finally:
        d_input.free()
        d_output.free()
        d_kernel.free()
        context.pop()
