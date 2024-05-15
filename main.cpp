#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/*
__global__ void hellocud(){
     std::cout << "Hello CUDA Func!"<<std::endl;
}*/

int main() {


    /*hellocud();

    hellocud<<<1, 1>>>();*/

    CUresult res = cuInit(0);

    std::cout << "Hello CUDA! Result: " << res << std::endl;

    return 0;
}

/*
Ekran çıktısı:
Hello CUDA! Result: 100 (100 ise hata kodu driver bulamadı hatası veriyor.)
*/