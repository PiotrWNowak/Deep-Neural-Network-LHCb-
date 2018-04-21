# Praca inżynierska

W niniejszej pracy zbudowano sztuczną sieć neuronową w języku C++, a następnie zaimplementowano wersję programu w architekturze CUDA. Celem było stworzenie wydajnej i poprawnie działającej sieci neuronowej bez wykorzystania gotowych bibliotek do klasyfikatora przypadków w eksperymencie LHCb. Na danych z symulacji Monte Carlo dokonano analizy działania sieci neuronowej w zależności od zastosowanych rozwiązań.


## Przed uruchomieniem

Aby skompilować program należy użyć komendy
```
nvcc -std=c++11 main.cu cpu.cu gpu.cu -o program_name
```

## Dane

Aby sieć neuronowa mogła rozwiązać problem należy przygotować dane do uczenia. W folderze "dane" jest przykładowy zbiór danych po normalizacji i podzielonych na dwa pliki. Jeden zawierający przypadki prawdziwe, drugi fałszywe. 

<p align="center">
  <img src="https://github.com/PiotrWNowak/Deep-Neural-Network-LHCb-/tree/master/analysys/images/1.png">
</p>

### Test

test

## Authors

**Piotr Nowak** - *Trenowanie sieci neuronowej do klasyfikacji śladów w eksprymencie LHCb przy wykorzystaniu kart graficznych* 
