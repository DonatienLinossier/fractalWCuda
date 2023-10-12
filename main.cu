
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <SDL.h>

#include <stdio.h>
#include <iostream>
#include "const.cpp"


float max_Iteration = MAX_ITERATION;
long double  min_reel =  - 2.0;
long double  max_reel =  2.0;
long double  min_imaginary =  - 2.0;
long double  max_imaginary =  2.0;
float centerX = 700;
float centerY = 500;

struct Complex {
    double real;
    double imag;
};

__device__ Complex add(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

__device__ Complex multiply(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

__device__ double magnitude(Complex z) {
    return sqrt(z.real * z.real + z.imag * z.imag);
}

__global__ void mandelbrot(uchar3* dev_gpuPixels, int width, int height, long double  min_reel, long double  max_reel, long double  min_imaginary, long double  max_imaginary, int max_Iteration) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= width * height) {
        return; // Out-of-bounds thread
    }
    int x = (blockIdx.x * blockDim.x + threadIdx.x) % width;
    int y = (blockIdx.x * blockDim.x + threadIdx.x) / width;

    double real = min_reel + (x * (max_reel - min_reel) / WIDTH);
    double imag = min_imaginary + (y * (max_imaginary - min_imaginary) / HEIGHT);



    Complex c = { real, imag};
    Complex z = { 0, 0 };

    int iterations = 0;

    while (magnitude(z) < 2.0 && iterations < max_Iteration) {
        Complex zSquared = multiply(z, z);
        z = add(zSquared, c);
        iterations+= 1;
    }

    if (iterations >= max_Iteration) {
        dev_gpuPixels[y * width + x] = { 0, 0, 0 };
    }
    else {
        dev_gpuPixels[y * width + x] = { static_cast<unsigned char>(R_MIN + (R_MAX - R_MIN) * R_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (R_MAX - R_MIN)),
                                         static_cast<unsigned char>(G_MIN + (G_MAX - G_MIN) * G_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (G_MAX - G_MIN)),
                                         static_cast<unsigned char>(B_MIN + (B_MAX - B_MIN) * B_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (B_MAX - B_MIN))};
    }
}


__global__ void juliaSet(uchar3* dev_gpuPixels, int width, int height, Complex number, long double  min_reel, long double  max_reel, long double  min_imaginary, long double  max_imaginary, int max_Iteration) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= width * height) {
        return; // Out-of-bounds thread
    }


    int x = (blockIdx.x * blockDim.x + threadIdx.x) % width;
    int y = (blockIdx.x * blockDim.x + threadIdx.x) / width;

    //dev_gpuPixels[y * width + x] = { 0, 255, 0};
    //return;

    double real = min_reel + (x * (max_reel - min_reel) / WIDTH);
    double imag = min_imaginary + (y * (max_imaginary - min_imaginary) / HEIGHT);
    //printf("\n[%f, %f]", real, imag);
    Complex z = { real, imag };

    int iterations = 0;

    while (magnitude(z) < 2.0 && iterations < max_Iteration) {
        Complex zSquared = multiply(z, z);
        z = add(zSquared, number);
        iterations+= 2;
    }


    if (iterations >= max_Iteration) {
        dev_gpuPixels[y * width + x] = { 0, 0, 0 };
    }
    else {
        //printf("%d", iterations);
        dev_gpuPixels[y * width + x] = { static_cast<unsigned char>(R_MIN + (R_MAX-R_MIN) * R_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (R_MAX - R_MIN)),
                                         static_cast<unsigned char>(G_MIN + (G_MAX - G_MIN) * G_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (G_MAX-G_MIN)),
                                         static_cast<unsigned char>(B_MIN + (B_MAX-B_MIN) * B_REVERSE - (double)(max_Iteration - iterations) / max_Iteration * (B_MAX - B_MIN))};
                                         
            
            //static_cast<unsigned char>(25 + (double)(max_Iteration-iterations) / max_Iteration * 25)};
    }
}

__global__ void gaussianBlurInPlace(uchar3* image, int width, int height)
{

    float gaussianKernel5x5[25] = {
        1.0f / 256,  4.0f / 256,  6.0f / 256,  4.0f / 256, 1.0f / 256,
        4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256,
        6.0f / 256, 24.0f / 256, 36.0f / 256, 24.0f / 256, 6.0f / 256,
        4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256,
        1.0f / 256,  4.0f / 256,  6.0f / 256,  4.0f / 256, 1.0f / 256
    };
    int kernelSize = 5;
    float* kernel = gaussianKernel5x5;
    int x = (blockIdx.x * blockDim.x + threadIdx.x) % width;
    int y = (blockIdx.x * blockDim.x + threadIdx.x) / width;
    if (x < width && y < height)
    {
        float3 result = make_float3(0.0f, 0.0f, 0.0f);

        for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
        {
            for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
            {
                int xOffset = x + i;
                int yOffset = y + j;

                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
                {
                    int kernelIndex = (i + kernelSize / 2) * kernelSize + (j + kernelSize / 2);
                    uchar3 pixel = image[yOffset * width + xOffset];
                    result.x += static_cast<float>(pixel.x) * kernel[kernelIndex];
                    result.y += static_cast<float>(pixel.y) * kernel[kernelIndex];
                    result.z += static_cast<float>(pixel.z) * kernel[kernelIndex];
                }
            }
        }

        image[y * width + x] = make_uchar3(static_cast<unsigned char>(result.x),
            static_cast<unsigned char>(result.y),
            static_cast<unsigned char>(result.z));
    }
}


void call_julia(uchar3* dev_gpuPixels, int width, int height, Complex number, long double  min_reel, long double  max_reel, long double  min_imaginary, long double  max_imaginary, int max_Iteration) {
    int nbthread = 1024;
    int numBlocks = (width * height + nbthread - 1) / nbthread;


    juliaSet << <numBlocks, nbthread >> > (dev_gpuPixels, width, height, number, min_reel, max_reel, min_imaginary, max_imaginary, max_Iteration);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("\ncall_collisionAoS - fin error: %s\n", cudaGetErrorString(cudaStatus));
    }

}

void call_mandelbrot(uchar3* dev_gpuPixels, int width, int height, long double  min_reel, long double  max_reel, long double  min_imaginary, long double  max_imaginary, int max_Iteration) {
    int nbthread = 1024;
    int numBlocks = (width * height + nbthread - 1) / nbthread;


    mandelbrot << <numBlocks, nbthread >> > (dev_gpuPixels, width, height, min_reel, max_reel, min_imaginary, max_imaginary, max_Iteration);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("\ncall_collisionAoS - fin error: %s\n", cudaGetErrorString(cudaStatus));
    }

}

void call_gaussianBlur(uchar3* dev_gpuPixels, int width, int height) {
    int nbthread = 1024;
    int numBlocks = (width * height + nbthread - 1) / nbthread;


    gaussianBlurInPlace << <numBlocks, nbthread >> > (dev_gpuPixels, WIDTH, HEIGHT);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("\ncall_collisionAoS - fin error: %s\n", cudaGetErrorString(cudaStatus));
    }

}


int getDisplayFromGpu(uchar3* hostPixels, uchar3* dev_gpuPixels, int width, int height) {

    cudaError_t err = cudaMemcpy(hostPixels, dev_gpuPixels, width * sizeof(uchar3) * height, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        printf("Erreur à l'interieur du blit: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int calculate_max_iter(double zoom_level, int initial_max_iter) {
    // Adjust max_iter based on the zoom level
    return static_cast<int>(initial_max_iter + log2(1.0 + zoom_level) * 20);
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow("SDL Example", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        return 2;
    }

    // Create a renderer
    SDL_Renderer* pRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!pRenderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        return 3;
    }
    SDL_Texture* pTexture = SDL_CreateTexture(pRenderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

    uchar3* dev_gpuPixels;
    uchar3* hostPixels;

    //Allocation of the Pixels on the gpu
    cudaMalloc(&dev_gpuPixels, WIDTH * sizeof(uchar3) * HEIGHT);

    cudaMallocHost(&hostPixels, WIDTH * HEIGHT * sizeof(uchar3));



    //printf("Done");

    bool quit = false;
    SDL_Event events;
    SDL_Point MousePosition;
    bool locked = true;
    Complex number = { 0.5, 0.5};
    while (!quit) {
        while (SDL_PollEvent(&events)) {
            switch (events.type)
            {
            case SDL_QUIT:
                quit = true;
                break;

            case SDL_MOUSEMOTION:
                SDL_GetMouseState(&MousePosition.x, &MousePosition.y);
                break;

            case SDL_MOUSEBUTTONUP:
                locked = !locked;

            case SDL_KEYDOWN :
                SDL_Keycode keyCode = events.key.keysym.sym;
                if (keyCode == SDLK_d) {
                    min_reel += (max_reel - min_reel) / 100;
                    max_reel += (max_reel - min_reel) / 100;
                }
                if (keyCode == SDLK_q) {
                    min_reel -= (max_reel - min_reel) / 100;
                    max_reel -= (max_reel - min_reel) / 100;
                }
                if (keyCode == SDLK_s) {
                    min_imaginary += (max_imaginary - min_imaginary) / 100;
                    max_imaginary += (max_imaginary - min_imaginary) / 100;
                }
                if (keyCode == SDLK_z) {
                    min_imaginary -= (max_imaginary - min_imaginary) / 100;
                    max_imaginary -= (max_imaginary - min_imaginary) / 100;
                }
                if (keyCode == SDLK_a) {
                    min_reel += (max_reel - min_reel) / 100;
                    max_reel -= (max_reel - min_reel) / 100;
                    min_imaginary += (max_imaginary - min_imaginary) / 100;
                    max_imaginary -= (max_imaginary - min_imaginary) / 100;
                }
                if (keyCode == SDLK_e) {
                    min_reel -= (max_reel - min_reel) / 100;
                    max_reel += (max_reel - min_reel) / 100;
                    min_imaginary -= (max_imaginary - min_imaginary) / 100;
                    max_imaginary += (max_imaginary - min_imaginary) / 100;
                }
                if (keyCode == SDLK_r) {
                    min_reel = -2.0;
                    max_reel = 2.0;
                    min_imaginary = -2.0;
                    max_imaginary = 2.0;
                }
            }
        }

        //Complex number = { 0.6, 0.5 };
        if(!locked)
            number = { -2 + (float) MousePosition.x / WIDTH * (2 - -2), -2 + (float) MousePosition.y / HEIGHT * (2 - -2) };

        double zoom_level = 1.0 / (fabs(max_reel - min_reel) < fabs(max_imaginary - min_imaginary) ?
            fabs(max_reel - min_reel) : fabs(max_imaginary - min_imaginary));
        //printf("\n %d", (int)calculate_max_iter(zoom_level, max_Iteration));
        if(MODE==0)
            call_julia(dev_gpuPixels, WIDTH, HEIGHT, number, min_reel, max_reel, min_imaginary, max_imaginary, (int)calculate_max_iter(zoom_level, max_Iteration));
        else if(MODE==1)
            call_mandelbrot(dev_gpuPixels, WIDTH, HEIGHT, min_reel, max_reel, min_imaginary, max_imaginary, (int)calculate_max_iter(zoom_level, max_Iteration));

        //call_gaussianBlur(dev_gpuPixels, WIDTH, HEIGHT);
        getDisplayFromGpu(hostPixels, dev_gpuPixels, WIDTH, HEIGHT);
        SDL_UpdateTexture(pTexture, NULL, hostPixels, WIDTH * sizeof(uchar3));
        SDL_RenderCopy(pRenderer, pTexture, NULL, NULL);
        SDL_RenderPresent(pRenderer);

    }
        SDL_DestroyRenderer(pRenderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        return 0;

}