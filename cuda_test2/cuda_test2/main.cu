#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <string>


class Sporcu {
private:
    double speed;
    double position;
    int id;

public:
    __device__ __host__ Sporcu() : id(-1), speed(0), position(0.0) {}

    __device__ __host__ Sporcu(int id, double speed) : id(id), speed(speed), position(0.0) {}

    __device__ __host__ void updatePosition(double time) {
        position += speed * time;
    }

    __device__ __host__ double getPosition() {
        return position;
    }

    __device__ __host__ int getId() {
        return id;
    }
};

class Takim {
private:
    Sporcu* sporcular;
    int maxSprocuSayisi = 3;
    int takimId;
    int athleteIndex;

public:
    __device__ __host__ Takim() : takimId(-1) {
		sporcular = nullptr;
        athleteIndex = 0;
    }

    __device__ __host__ Takim(int teamId, int max) : takimId(teamId), maxSprocuSayisi(max){
        sporcular = new Sporcu[maxSprocuSayisi];
        athleteIndex = 0;
    }

    __device__ __host__ void addAthlete(Sporcu& athlete) {
		if (athleteIndex < 3) {
			sporcular[athleteIndex++] = athlete;
		}
		else {
			printf("Takimda yer yok\n");
		}
	}

    __device__ __host__ Sporcu* getAthletes() {
        return sporcular;
    }
};


__global__ void simulateRaceKernel(Sporcu* athletes, int numAthletes, double time) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("idx %d\n",idx);
    if (idx < numAthletes) {
        athletes[idx].updatePosition(time);
    }
}

int main() {

    int takimSayisi = 100;
    int birTakimdakiSporcuSayisi = 3;
    int toplamSporcuSayisi = takimSayisi * birTakimdakiSporcuSayisi;

    int blockSize = 400;
    int numBlocks = (toplamSporcuSayisi + blockSize - 1) / blockSize;


    //Takimlar ve sprocular olusturulur.
    Sporcu* athletesHost;
    cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

    Takim* teamsHost;
    cudaMallocHost((void**)&teamsHost, 100 * sizeof(Takim));

    for(int i = 0; i < 100; i++) {
        int team_id = i+1;
		teamsHost[i] = Takim(team_id, birTakimdakiSporcuSayisi);
        for (int j = 0; j < birTakimdakiSporcuSayisi; j++) {
            int athleteIndex = i * birTakimdakiSporcuSayisi + j;
            int athleteId = i * birTakimdakiSporcuSayisi + j + 1;
            athletesHost[athleteIndex] = Sporcu(athleteId, (double)rand() / RAND_MAX * 5.0 + 1.0);

            teamsHost[i].addAthlete(athletesHost[athleteIndex]);
            std::cout<<"Athlete " << athleteId << " added to team " << i << std::endl;
        }
	}




    //Paralel hesaplama icin sporcularin GPU kopyasi olusturulur.
    Sporcu* athletesDevice;
    cudaMalloc((void**)&athletesDevice, toplamSporcuSayisi * sizeof(Sporcu));
    cudaMemcpy(athletesDevice, athletesHost, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyHostToDevice);

    for (int i = 0; i < 100; i++) { // simulate 100 seconds
        simulateRaceKernel << <numBlocks, blockSize >> > (athletesDevice, toplamSporcuSayisi, 1.0);
        cudaDeviceSynchronize();

        // print updated positions
        cudaMemcpy(athletesHost, athletesDevice, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyDeviceToHost);
        for (int j = 0; j < toplamSporcuSayisi; j++) {
            std::cout << "Athlete " << athletesHost[j].getId() << ": " << athletesHost[j].getPosition() << " meters" << std::endl;
        }
    }

    cudaFreeHost(athletesHost);
    cudaFree(athletesDevice);



    return 0;
}
    
   
 
