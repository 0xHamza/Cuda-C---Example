# Cuda C++ Example

## Triatlon Yarışı Projesi


Bu proje, Ubuntu üzerinde OOP (Object-Oriented Programming) ve CUDA CPP (Compute Unified Device Architecture C++) kullanarak 300 takım ve her takımda 3 sporcu bulunan bir triatlon yarışını her yarışmacı için eş zamanlı simüle etmeyi amaçlamaktadır.


## Sistem Özellikleri


1. Öncelikle VmWare üzerinden Ubuntu ya Cuda yı kurmaya çalıştım 'nvcc --version' ile cuda versiyonu geliyor ancak, 'nvidia-smi' komutunu çalıştırınca hiç bir türlü nvidia driverlerını bulamıyordu ve yüklü değil diye hata veriyordu 'bash' de güncelleme uzun saatler kurmayı çalışmama rağmen problem çözülmedi.

- VmWare Sanal Makine üzerinde çalışan Ubuntu 22.04
- Ram 4gb
- İşlemci çekirdeği 2
- GTX 1050


2. Daha sonrasında windows üzerinde kurulu WSL 2 ile ubuntu 22.04 çalıştırarak aynı işlemleri tekrar denedim ve en sonunda gen aynı vmware de olan problem ile karşılaştım. Anlaşılan sanal makineler için ekstra nvidia driver ayarlamaları gerekiyor.

3. İlk iki yöntem ile kuramayınca bilgisayarıma dual boot olarak ubuntu kurmayı düşündüm ancak çok fazla zaman kalmadığı için proje teslimine, sıfırdan ubuntu kurarsam sistem güncellemeleri driver güncellemeri derken gene zaman kaybedecektim. 
Bu yüzden Windows bilgisayarıma kurararak ilerlemeye en azından çalışan bir proje çıkarmaya başladım.

Kullanılan işletim sistemi:

- GTX1050
- 20GB Ram
- Visual Studio 2022 
- NVIDIA-SMI 551.78                 Driver Version: 551.78         CUDA Version: 12.4

## Proje Adımları

- [x] Proje Yapısını Hazırlama
  - [x] ~Ubuntu 22.04~ windows 10 Cuda kurulumlarını ve ayarları.
  - [ ] Vs2022 cuda projesi oluşturma C++ proje linker ve lib o
gerekli kaynak dosyalarının eklenmesi.


- [] Sınıf Yapısını Oluşturma
  - [x] `Sporcu` sınıfı: Sporcuların konum, hız ve diğer özelliklerini tutan bir sınıf
  - [x] `Takim` sınıfı: Takımları temsil eden bir sınıf
  - [x] `Triatlon` sınıfı: Triatlon yarışını yöneten ana sınıf



## eklenen diğer yenilikler

1. **Triathlon Sınıfı**:
   - Bu sınıf, ana triatlon yarışının yönetiminden sorumludur.
   - Yarış ayarlama ve başlatma işlemlerini gerçekleştirir.
   - Takımları ve sporcuları ekleyebilir.
   - Yarış süresince sporcuların konumlarını ve hızlarını güncelleyebilir.

2. **Parkur Sınıfı**:
   - Triathlon yarışının üç farklı disiplinini (yüzme, bisiklet, koşu) temsil eder.
   - Her parkur, zorluğuna göre bir güçlük faktörüne sahiptir.
   - Sporcuların hızları, parkur zorluğuna göre artırılır.

3. **Takım Sınıfı**:
   - Her takım, 3 sporcudan oluşur.
   - Takım içindeki sporcuların bilgilerini (konum, hız, vb.) tutar.
   - Takımların toplam sürelerini hesaplar.

4. **Sporcu Sınıfı**:
   - Sporcuların konum, hız ve diğer özelliklerini temsil eder.
   - Sporcuların parkurlardaki performanslarını simüle eder.

5. **CUDA Fonksiyonları**:
   - `yarışıAyarla()`: Triathlon yarışını ayarlar, takımları ve sporcuları ekler.
   - `yarışıBaşlat()`: Yarışı başlatır, sporcuların hızlarını ve konumlarını güncelleyerek her saniye simüle eder.
   - Bu fonksiyonlar, CUDA iş parçacıkları (threads) kullanılarak paralel olarak çalıştırılacaktır.

## Proje Adımları

1. **Proje Yapısını Hazırlama**:
   - Windows 10 işletim sistemi üzerinde Visual Studio 2022 kullanılacak.
   - CUDA kurulumu ve ayarları yapılacak.
   - C++ proje oluşturulacak, gerekli kaynak dosyaları eklenecek.

2. **Sınıf Yapısını Oluşturma**:
   - `Sporcu` sınıfı: Sporcuların konum, hız ve diğer özelliklerini tutar.
   - `Takim` sınıfı: Takımları temsil eder, takım içindeki sporcuların bilgilerini tutar.
   - `Parkur` sınıfı: Triathlon yarışının üç farklı disiplinini (yüzme, bisiklet, koşu) temsil eder.
   - `Triatlon` sınıfı: Ana triatlon yarışını yönetir, yarış ayarlama ve başlatma işlemlerini gerçekleştirir.

3. **CUDA Fonksiyonlarının Geliştirilmesi**:
   - `yarışıAyarla()` fonksiyonu: Takımları ve sporcuları ekler, yarış ayarlarını yapar.
   - `yarışıBaşlat()` fonksiyonu: Yarışı başlatır, her saniye sporcuların konumlarını ve hızlarını güncelleyerek simülasyonu gerçekleştirir.
