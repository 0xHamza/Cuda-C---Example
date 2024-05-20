#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>

class Parkur {
private:
    double uzunluk;
    double zorluk;
    std::string name;

public:
    __device__ __host__ Parkur(double uzunluk, double zorluk, const std::string& name) : uzunluk(uzunluk), zorluk(zorluk), name(name) {}

    __device__ __host__ double getUzunluk() {
        return uzunluk;
    }

    __device__ __host__ double getZorluk() {
        return zorluk;
    }

    __device__ __host__ const std::string& getName() {
        return name;
    }
};

class Sporcu {
private:
    double speed;
    double position;
    int id;
    int teamId;

public:
    __device__ __host__ Sporcu() : id(-1), speed(0), position(0.0) {}

    __device__ __host__ Sporcu(int id, double speed) : id(id), speed(speed), position(0.0) {}

    __device__ __host__ Sporcu(int id, int tid, double speed) : id(id), teamId(tid), speed(speed), position(0.0) {}

    __device__ __host__ void updatePosition(double time) {
        position += speed * time;
    }

    std::string getSporcuBilgi() {
		
        return "Sporcu " + std::to_string(id) + " speed " + std::to_string(speed) + " distance " + std::to_string(position);
	}

    __device__ __host__ double getPosition() {
        return position;
    }

    __device__ __host__ int getId() {
        return id;
    }

    __device__ __host__ double getSpeed() {
        return speed;
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

    Sporcu* getSporcular() const {
		return sporcular;
	}

    Sporcu* getSporcularAdress() const {
        return sporcular;
    }

    __device__ __host__ int getMaxSprocuSayisi() {
        return maxSprocuSayisi;
    }
};

class Takim2 {
private:
    std::vector<Sporcu> sporcular;
    int maxSporcuSayisi;
    int takimId;

public:
    Takim2() : maxSporcuSayisi(1000), takimId(-1) {}

    Takim2(int teamId, int max) : maxSporcuSayisi(max), takimId(teamId) {
        sporcular.reserve(max);
    }

    void addAthlete(const Sporcu& athlete) {
        if (sporcular.size() < maxSporcuSayisi) {
            sporcular.push_back(athlete);
        }
        else {
            std::cout << "Takimda yer yok" << std::endl;
        }
    }

    void printSporcular() {
		for (Sporcu sporcu : sporcular) {
			std::cout << "tid "<<takimId<<": Sporcu " << sporcu.getSporcuBilgi() << std::endl;
		}
	}

    const std::vector<Sporcu>& getSporcular() const {
        return sporcular;
    }

    void setSporcular(const std::vector<Sporcu>& sporcular) {
		this->sporcular = sporcular;
	}

    int getMaxSprocuSayisi() {
		return maxSporcuSayisi;
	}

	int getTakimId() {
		return takimId;
	}
};

__global__ void simulateRaceKernel(Sporcu* athletes, int numAthletes, double time, int *gosterilecekler, int gosterileceklerSayisi) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < numAthletes) {        
        athletes[threadId].updatePosition(time);
        //printf("%d: Sporcu  %d: %0.2fm/s \t %0.2fsn \t %0.2f metre\n", blockIdx.x, athletes[threadId].getId(), athletes[threadId].getSpeed(), time, athletes[threadId].getPosition());
        
        for (int i = 0; i < gosterileceklerSayisi; i++)
        {
            if (athletes[threadId].getId() == gosterilecekler[i])
                printf("%d: Sporcu  %d: %0.2fm/s \t %0.2fsn \t %0.2f metre\n", blockIdx.x, athletes[threadId].getId(), athletes[threadId].getSpeed(), time, athletes[threadId].getPosition());
        }
        
        
    }
}



class Triatlon {
private:
    Parkur* parkurlar;
    Takim* takimlar;
    Takim2* takimlar2;

    std::vector<Parkur> parkur2_v;

    std::vector<Takim2> takimlar2_v;

    int takimSayisi         = 1;
    int parkurSayisi        = 1;
    int toplamSporcuSayisi  = 1;

public:
    __device__ __host__ Triatlon(Parkur* parkurlar, int parkurSayisi, Takim* takimlar, int takimSayisi) 
        : parkurSayisi(parkurSayisi), parkurlar(parkurlar), takimlar(takimlar), takimSayisi(takimSayisi) {
        toplamSporcuSayisi = takimSayisi * takimlar->getMaxSprocuSayisi();
    }

    __host__ Triatlon(Parkur* parkurlar, int parkurSayisi, Takim2* takimlar, int takimSayisi)
        : parkurSayisi(parkurSayisi), parkurlar(parkurlar), takimlar2(takimlar), takimSayisi(takimSayisi) {
        
        int total = 0;
        
        for (int i = 0; i < takimSayisi; i++)
            total+= takimlar[i].getSporcular().size();
        
        toplamSporcuSayisi = total;
    }

    

    __device__ __host__ Triatlon(std::vector<Parkur>  parkurlar, int parkurSayisi, std::vector<Takim2> takimlar, int takimSayisi)
        : parkurSayisi(parkurSayisi), parkur2_v(parkurlar), takimlar2_v(takimlar), takimSayisi(takimSayisi) {
        //total number of athletes for vector takimlar2
        int total = 0;
        for (Takim2 _takim2: takimlar2_v)
        {
            total = total + _takim2.getSporcular().size();
        }

        toplamSporcuSayisi = total;

        std::cout<<"Triatlona dahil edilen yarismaci sayisi: "<<total<<std::endl;
    }


    __device__ __host__ void yarisiBaslat() {
        for (int i = 0; i < parkurSayisi; i++) {
            printf("Starting race on %s\n", parkurlar[i].getName());
            
            for (int j = 0; j < takimlar->getMaxSprocuSayisi(); j++) {
                Sporcu* sporcular = takimlar->getSporcular();
                sporcular[j].updatePosition(parkurlar[i].getUzunluk() / sporcular[j].getSpeed());
            }
        }
    }

    __host__ int getTakimSayisi() {
		return takimSayisi;
    }

    __host__ int getParkurSayisi() {
        return parkurSayisi;
	}

    __host__ int getToplamSporcuSayisi() {
		    return toplamSporcuSayisi;
    }

    __host__ int getTakimMaxSporcuSayisi() {
		return takimlar->getMaxSprocuSayisi();
    }

    __host__ void yarisiBaslatGPU() {

        std::cout<<"Parkur sayisi: "<< getParkurSayisi()<<std::endl;
        std::cout<<"Takim sayisi: "<< getTakimSayisi() <<std::endl;
        std::cout<<"Toplam sporcu sayisi: "<< getToplamSporcuSayisi() <<std::endl;


        Sporcu* athletesHost;
        cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

        int index = 0;
        //GPU GONDERILECEK Sporculari cek
        for (int i = 0; i < takimSayisi; i++) {
            
            std::cout << "Takim " << takimlar2[i].getTakimId() << ", " << takimlar2[i].getSporcular().size() << std::endl;
            
            const std::vector<Sporcu>& sporcular = takimlar2[i].getSporcular();

           

            for (auto sporcu : sporcular) {
                athletesHost[index++] = sporcu;
                std::cout << "\t::Sporcu " << sporcu.getId() << " speed " << sporcu.getSpeed() << std::endl;
            }
        }

        
        std::string gosterilecekler;

        // User input ile gösterilecek takım üyelerinin indekslerini alın
        std::cout << "Hangi takim uyeleri hesaplama sirasinda ekranda gosterilsin? (Örnek girdi: 0 1 2): ";
        std::getline(std::cin, gosterilecekler);

        //split gosterilecekler
   
        std::istringstream iss(gosterilecekler);
        int number;
        int gosterilecekTeamsSayisi = 0;
        int* gosterilecekTeams = nullptr;
        while (iss >> number) {
            gosterilecekTeams = (int*)realloc(gosterilecekTeams, (gosterilecekTeamsSayisi + 1) * sizeof(int));
            gosterilecekTeams[gosterilecekTeamsSayisi] = number;
            gosterilecekTeamsSayisi++;
        }

        int* gosterileceklerDevice;
        cudaMalloc((void**)&gosterileceklerDevice, gosterilecekTeamsSayisi * sizeof(int));
        cudaMemcpy(gosterileceklerDevice, gosterilecekTeams, gosterilecekTeamsSayisi * sizeof(int), cudaMemcpyHostToDevice);

        Sporcu* athletesDevice;
        cudaMalloc((void**)&athletesDevice, toplamSporcuSayisi * sizeof(Sporcu));
        cudaMemcpy(athletesDevice, athletesHost, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyHostToDevice);
            

       
        // CUDA çekirdeğini çağır
        int blockSize = 256;
        int blockCount = (toplamSporcuSayisi + blockSize - 1) / blockSize;



        // 100 saniye simüle et
        for (int i = 0; i < 10; i++) { 
            simulateRaceKernel<<<blockCount, blockSize >>>(athletesDevice, toplamSporcuSayisi, i, gosterileceklerDevice, gosterilecekTeamsSayisi);
            cudaDeviceSynchronize();
        }

        // Sonuçları cihazdan ana belleğe geri kopyala
        cudaMemcpy(athletesHost, athletesDevice, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyDeviceToHost);

        //print sprocular gpu
        for (int i = 0; i < toplamSporcuSayisi; i++) {
            std::cout << ":"<<i<<":Sporcu " << athletesHost[i].getSporcuBilgi()<<std::endl;
        }

       

        // Kaynakları serbest bırak
        cudaFree(athletesDevice);
        
    }

};

int main() {

    int takimSayisi = 100;
    int birTakimdakiSporcuSayisi = 3;
    int toplamSporcuSayisi = takimSayisi * birTakimdakiSporcuSayisi;

    int blockSize = 200;  //GTX1050 icin max 640 thread, 
    int blockCount = (toplamSporcuSayisi + blockSize - 1) / blockSize;

    Parkur yuzme    (50000,      1,      "P_Yuzme"   );
    Parkur bisiklet (400000,     3,      "P_Bisiklet");
    Parkur kosu     (10000,      1/3,    "P_Kosu"    );

    Parkur parkurlar[] = { yuzme, bisiklet, kosu };

    //Takimlar ve sprocular olusturulur.
    Sporcu* athletesHost;
    cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

    Takim2* teamsHost;
    cudaMallocHost((void**)&teamsHost, takimSayisi * sizeof(Takim2));
    

    for(int i = 0; i < 100; i++) {
        int team_id = i+1;
		teamsHost[i] = Takim2(team_id, birTakimdakiSporcuSayisi);
        for (int j = 0; j < birTakimdakiSporcuSayisi; j++) {
            int athleteIndex = i * birTakimdakiSporcuSayisi + j;
            int athleteId = i * birTakimdakiSporcuSayisi + j + 1;
            athletesHost[athleteIndex] = Sporcu(athleteId, team_id, (float)rand() / RAND_MAX * 4.0 + 1.0);

            teamsHost[i].addAthlete(athletesHost[athleteIndex]);
            std::cout<<"::Sporcu " << athleteId << " speed " << athletesHost[athleteIndex].getSpeed() << " added to team " << i << std::endl;
        }
	}


    //Triatlon for CPU
    Triatlon* triatlonHost;
    cudaMallocHost((void**)&triatlonHost, sizeof(Triatlon));

    //triatlon add parkurlar
    triatlonHost = new Triatlon(parkurlar, 3, teamsHost, takimSayisi);

    std::cout << "Takimlar olusturuldu\n";
    triatlonHost->yarisiBaslatGPU();

      


    cudaFreeHost(athletesHost);
    cudaFreeHost(teamsHost);
    cudaFreeHost(triatlonHost);

    return 0;
}
