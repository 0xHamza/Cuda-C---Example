
#include <iostream>
#include <cuda.h>

int main(){

    CUresult res = cuInit(0);
    std::cout<<"Hello cuda! : "<<res<std::endl;

    return 0;
}