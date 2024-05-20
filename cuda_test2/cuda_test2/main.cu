#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#include <curand_kernel.h>
#include <sstream>



class Parkur {
private:
    double uzunluk;
    double zorluk;
    std::string name;

public:

    __device__ __host__ Parkur() : uzunluk(0.0), zorluk(0.0) {}

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
    Parkur* parkurlar;
    int currentParkur = 0;
    int sonSure = 0;
    bool bekleme = false;
    int beklemeSuresi = 10;
    int toplamParkurSayisi = 3;
    int alinanYol = 0;
    int guncelSure = 0;

public:

    int bilgiyiGoruntule = 0;

    __device__ __host__ Sporcu() : id(-1), speed(0), position(0.0) {}

    __device__ __host__ Sporcu(int id, double speed) : id(id), speed(speed), position(0.0) {}

    __device__ __host__ Sporcu(int id, int tid, double speed) : id(id), teamId(tid), speed(speed), position(0.0) {}

    __device__ __host__ void yarisiAyarla(Parkur* parkurlars, int *gosterilecekler, int gosterilecekSayisi, double spd) {
	

        //printf("%d: adet %d s %f : %d-%d \n",id,gosterilecekSayisi, spd, gosterilecekler[0], gosterilecekler[1]);
        
        parkurlar = parkurlars;
        speed = spd;
        currentParkur = 0;

        for (int i = 0; i < gosterilecekSayisi; i++)
        {
            if (id == gosterilecekler[i]) {
                bilgiyiGoruntule = 1;
            }
        }

        //printf("%d= self  0: %f,  1: %f,  2: %f -- %f -- %d\n", id, parkurlar[0].getUzunluk(), parkurlar[1].getUzunluk(), parkurlar[2].getUzunluk(),speed,bilgiyiGoruntule);
	}

    __device__ __host__ void bilgiprint() {
        //printf("teamid %d-%d, speed %f self %d pos %f > 0: %f,  1: %f,  2: %f\n", teamId, id, speed, bilgiyiGoruntule, position, parkurlar[0].getUzunluk(), parkurlar[1].getUzunluk(), parkurlar[2].getUzunluk());
        printf("%d: Sporcu  %d: %0.2fm/s \t %dsn \t %0.2f metre\t parkur %d, alinan_yol %d\n", teamId, id, parkurlar[currentParkur].getZorluk() * speed, guncelSure-sonSure, position,currentParkur,alinanYol);
    }



    __device__ __host__ int updatePosition(double time) {
        
        double sporcuParkurHizi = parkurlar[currentParkur].getZorluk() * speed;
        
        guncelSure = time;

        if (bekleme == false)
        {
            if ( (sporcuParkurHizi * (time-sonSure)) >= (parkurlar[currentParkur].getUzunluk())) {
                
                
                alinanYol += parkurlar[currentParkur].getUzunluk();
                position = parkurlar[currentParkur].getUzunluk();
                bekleme = true;
                sonSure = time;
             
                if (currentParkur + 1 == toplamParkurSayisi) {      //yarış bitti
                   
                    return -3;
                }
                else{   							   //parkur bitti       
                    return -5;
                }
            }
            else {                                  //parkur devam ediyor
                position = (sporcuParkurHizi * (time - sonSure));
                
                return -1;
            }
        }
        else {
            if (time - sonSure >= beklemeSuresi) { //bekleme suresi bitti
                bekleme = false;
                sonSure = time;
                position = 0;
                currentParkur++;
                return -4;
            }
            else {                                  //bekleme suresi devam ediyor
                return -2;
            }
        }
    }

    std::string getSporcuBilgi() {
		
        return "Sporcu " + std::to_string(id) + " speed " + std::to_string(speed) + " distance " + std::to_string(position)+ " ::"+ std::to_string(bilgiyiGoruntule);
	}

    __device__ __host__ double getPosition() {
        return position;
    }

    __device__ __host__ int getId() {
        return id;
    }

    __device__ __host__ int getTeamId() {
        return teamId;
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

    void setSporcular(Sporcu* sporcular) {
        this->sporcular = sporcular;
	}

    void setSporcular(const std::vector<Sporcu>& sporcular) {
        for (int i = 0; i < sporcular.size(); i++) {
            this->sporcular[i] = sporcular[i];
        }
	}

    __device__ __host__ int getMaxSprocuSayisi() {
        return maxSprocuSayisi;
    }
};

class Takim2 {
private:
  
    int maxSporcuSayisi;
    int takimId;

public:
    std::vector<Sporcu> sporcular;

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

    void printAllSporcular() {
		for (Sporcu sporcu : sporcular) {
			std::cout << "\ttid " << takimId << ": Sporcu " << sporcu.getSporcuBilgi() << std::endl;
		}
	}

    int getMaxSprocuSayisi() {
		return maxSporcuSayisi;
	}

	int getTakimId() {
		return takimId;
	}
};



__global__ void simulateRaceKernel(Sporcu* athletes, int numAthletes, double time, int* yarisbittimi) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    

    if (threadId < numAthletes) {
        int durum = athletes[threadId].updatePosition(time);
        //printf("%d: Sporcu  %d: %0.2fm/s \t %0.2fsn \t %0.2f metre\n", blockIdx.x, athletes[threadId].getId(), athletes[threadId].getSpeed(), time, athletes[threadId].getPosition());


        if (durum == -1 && athletes[threadId].bilgiyiGoruntule) {
            athletes[threadId].bilgiprint();
        }

        if (durum == -2 && athletes[threadId].bilgiyiGoruntule) {
            printf("BEKLEME :: time %d\n", (int)time);
        }

        if (durum == -4 && athletes[threadId].bilgiyiGoruntule) {
            printf("BEKLEME BITTI ::time %d\n", (int)time);
        }

        if(durum == -5 && athletes[threadId].bilgiyiGoruntule) {
			printf("PARKUR BITTI ::time %d\n", (int)time);
		}

        if (durum == -3 && athletes[threadId].bilgiyiGoruntule) {
            athletes[threadId].bilgiprint();
            printf("\nYarisi bitirdi, %d: Sporcu  %d: %0.2fm/s \t %0.2fsn \t %0.2f metre\n", blockIdx.x, athletes[threadId].getId(), athletes[threadId].getSpeed(), time, athletes[threadId].getPosition());
            *yarisbittimi = 1;
        }
    }
}



__global__ void ayarlaRaceKernel(Sporcu* athletes, Parkur* parkurlar, int numAthletes, int* gosterilecekler, int gosterileceklerSayisi) {

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(clock(), threadId, 0, &state);

    // 1 ile 5 arasında rastgele bir sayı üret
    double speed = curand_uniform(&state) * 5.0 + 1.0;


    if (threadId < numAthletes)
    {
        athletes[threadId].yarisiAyarla(parkurlar, gosterilecekler, gosterileceklerSayisi, speed);
        if(athletes[threadId].bilgiyiGoruntule) athletes[threadId].bilgiprint();
    }
}

class Triatlon {
private:
    Parkur* parkurlar;
    Takim* takimlar;
    Takim2* takimlar2;

    std::vector<Parkur> parkur2_v;

    std::vector<Takim2> takimlar2_v;


    Sporcu *sprocuHost;
    Sporcu *sporcuDevice;

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

        /*
        Sporcu* athletesHost;
        cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

        int index = 0;
        //GPU GONDERILECEK Sporculari cek
        for (int i = 0; i < takimSayisi; i++) {
            
            std::cout << "Takim " << takimlar2[i].getTakimId() << ", " << takimlar2[i].getSporcular().size() << std::endl;
            
            const std::vector<Sporcu>& sporcular = takimlar2[i].getSporcular();

            for (auto sporcu : sporcular) {
                athletesHost[index++] = sporcu;
                //std::cout << "\t::Sporcu " << sporcu.getId() << " speed " << sporcu.getSpeed() << std::endl;
            }
        }

        

     
        Sporcu* athletesDevice;
        cudaMalloc((void**)&athletesDevice, toplamSporcuSayisi * sizeof(Sporcu));
        cudaMemcpy(athletesDevice, athletesHost, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyHostToDevice);
        
         */

       
        // CUDA çekirdeğini çağır
        int blockSize = 256;
        int blockCount = (toplamSporcuSayisi + blockSize - 1) / blockSize;



        int* yarisbittiMiDevice = 0;
        int yarisbittiMi = 0;

        // Bellekte yer ayırma
        cudaMalloc((void**)&yarisbittiMiDevice, sizeof(int));
        cudaMemcpy(yarisbittiMiDevice, &yarisbittiMi, sizeof(int), cudaMemcpyHostToDevice);

        /*
       //print sprocular gpu
        for (int i = 0; i < toplamSporcuSayisi; i++) {
			std::cout << ":"<<i<<":Sporcu " << sprocuHost[i].getSporcuBilgi()<<std::endl;
		}
        */



        // 100 saniye simüle et
        for (int i = 0; yarisbittiMi == 0; i++) {
            simulateRaceKernel <<< blockCount, blockSize >> > (sporcuDevice, toplamSporcuSayisi, i, yarisbittiMiDevice);
            cudaDeviceSynchronize();
            cudaMemcpy(&yarisbittiMi, yarisbittiMiDevice, sizeof(int), cudaMemcpyDeviceToHost);
            if(yarisbittiMi == 1) break;
        }

        // Bellekte ayrılan alanları serbest bırakma
        cudaFree(yarisbittiMiDevice);

        // Sonuçları cihazdan ana belleğe geri kopyala
        cudaMemcpy(sprocuHost, sporcuDevice, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyDeviceToHost);


        std::cout<<"Yarisi baslat GPU bitti"<<std::endl;
        //print sprocular gpu
        for (int i = 0; i < toplamSporcuSayisi; i++) {
            if(sprocuHost[i].bilgiyiGoruntule)
                std::cout << ":"<<i<<":Sporcu " << sprocuHost[i].getSporcuBilgi()<<std::endl;
        }

        // Kaynakları serbest bırak
        cudaFree(sporcuDevice);
        
    }


    __host__ void yarisiAyarlaGPU() {

        std::cout << "Parkur sayisi: " << getParkurSayisi() << std::endl;
        std::cout << "Takim sayisi: " << getTakimSayisi() << std::endl;
        std::cout << "Toplam sporcu sayisi: " << getToplamSporcuSayisi() << std::endl;


        Sporcu* athletesHost;
        cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

        int index = 0;
        //GPU GONDERILECEK Sporculari cek
        for (int i = 0; i < takimSayisi; i++) {
          
            const std::vector<Sporcu>& sporcular = takimlar2[i].getSporcular();

            for (auto sporcu : sporcular) {
                athletesHost[index++] = sporcu;
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


        //parkurlar device  
        Parkur* parkurlarDevice;
        cudaMalloc((void**)&parkurlarDevice, parkurSayisi * sizeof(Parkur));
        cudaMemcpy(parkurlarDevice, parkurlar, parkurSayisi * sizeof(Parkur), cudaMemcpyHostToDevice);


        // CUDA çekirdeğini çağır
        int blockSize = 256;
        int blockCount = (toplamSporcuSayisi + blockSize - 1) / blockSize;


         ayarlaRaceKernel << < blockCount, blockSize >> > (athletesDevice, parkurlarDevice, toplamSporcuSayisi, gosterileceklerDevice, gosterilecekTeamsSayisi);
          cudaDeviceSynchronize();
    

         std::cout<< "Yarisi ayarla kernel tamamlandi."<<std::endl;

        // Sonuçları cihazdan ana belleğe geri kopyala
        cudaMemcpy(athletesHost, athletesDevice, toplamSporcuSayisi * sizeof(Sporcu), cudaMemcpyDeviceToHost);

        sporcuDevice = athletesDevice;
        sprocuHost = athletesHost;

        // athletesHost to teamsHost
        for (int i = 0; i < toplamSporcuSayisi; i++) {
            // athletes to self team
            // takimlar2.sporcular std vector sporcu turundedir, guncelleme yapilacak olan sporcu eskisinin yerini almali
            std::vector<Sporcu> spr = takimlar2[athletesHost[i].getTeamId()].getSporcular();
            for (int j = 0; j < spr.size(); j++) {
                if (spr[j].getId() == athletesHost[i].getId()) {
                    takimlar2[athletesHost[i].getTeamId()].sporcular[j] = athletesHost[i];
                    break; // Güncelleme yapıldıktan sonra döngüyü sonlandırın
                }
            }
        }

        /*
        //print all takimlar2 sporcular
        for (int i = 0; i < takimSayisi; i++) {

			std::cout << "Takim " << takimlar2[i].getTakimId() << ", " << takimlar2[i].getSporcular().size() << std::endl;
			takimlar2[i].printAllSporcular();
		}
        */

        std::cout<<"Yarisi ayarla GPU bitti"<<std::endl;
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
    Parkur kosu     (10000,      0.33,    "P_Kosu"    );

    Parkur parkurlar[] = { yuzme, bisiklet, kosu };

    //Takimlar ve sprocular olusturulur.
    Sporcu* athletesHost;
    cudaMallocHost((void**)&athletesHost, toplamSporcuSayisi * sizeof(Sporcu));

    Takim2* teamsHost;
    cudaMallocHost((void**)&teamsHost, takimSayisi * sizeof(Takim2));
    

    for(int i = 0; i < 100; i++) {
        int team_id = i;
		teamsHost[i] = Takim2(team_id, birTakimdakiSporcuSayisi);
        for (int j = 0; j < birTakimdakiSporcuSayisi; j++) {
            int athleteIndex = i * birTakimdakiSporcuSayisi + j;
            int athleteId = i * birTakimdakiSporcuSayisi + j + 1;
            athletesHost[athleteIndex] = Sporcu(athleteId, team_id, (float)rand() / RAND_MAX * 4.0 + 1.0);

            teamsHost[i].addAthlete(athletesHost[athleteIndex]);
            //std::cout<<"::Sporcu " << athleteId << " speed " << athletesHost[athleteIndex].getSpeed() << " added to team " << i << std::endl;
        }
	}


    //Triatlon for CPU
    Triatlon* triatlonHost;
    cudaMallocHost((void**)&triatlonHost, sizeof(Triatlon));

    //triatlon add parkurlar
    triatlonHost = new Triatlon(parkurlar, 3, teamsHost, takimSayisi);

    std::cout << "Takimlar olusturuldu\n";
    triatlonHost->yarisiAyarlaGPU();
    triatlonHost->yarisiBaslatGPU();

      

    cudaFreeHost(athletesHost);
    cudaFreeHost(teamsHost);
    cudaFreeHost(triatlonHost);

    return 0;
}
