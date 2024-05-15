#include <iostream>
#include <cuda.h>
#include "device_launch_parameters.h"


__global__ void hellocud() {
    //fprintf(stderr, "cudaDeviceReset failed!");
    //std::cout << "Hello CUDA! FUNCCC!" << std::endl;
    printf("Hello CUDA Func!\n");
    //cudaPrintfDisplay("cudaPrintfDisplay");
}

void launchKernel() {
    hellocud<<<2, 2>>>();
    cudaDeviceSynchronize(); // Çekirdek fonksiyonun tamamlanmasını bekle
}

int main() {
    
    launchKernel();

    CUresult res = cuInit(0);
    std::cout << "Hello CUDA! Result: " << res << std::endl;


    return 0;
}

/*
Ekran çıktısı:
Hello CUDA! Result: 100
*/
