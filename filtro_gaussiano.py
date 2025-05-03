import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from time import time

cuda.init() 

def filtro_gaussiano_cuda(mascara, image):

    device = cuda.Device(0)
    context = device.make_context() 

    try:
        # Kernel CUDA
        mod = SourceModule("""
        """)

        #return output_image, bloques, hilos, duration
        
    finally:
        #d_input.free()
        #d_output.free()
        #d_kernel.free()
        context.pop()