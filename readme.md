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

- [ ] Proje Yapısını Hazırlama
  - [ ] Ubuntu 22.04 e Cuda kurulumlarını ve ayarları.
  - [ ] C++ proje klasörünü oluşturun ve gerekli kaynak dosyalarının eklenmesi.


- [ ] Sınıf Yapısını Oluşturma
  - [ ] `Sporcu` sınıfı: Sporcuların konum, hız ve diğer özelliklerini tutan bir sınıf
  - [ ] `Takim` sınıfı: Takımları temsil eden bir sınıf
  - [ ] `Triatlon` sınıfı: Triatlon yarışını yöneten ana sınıf

