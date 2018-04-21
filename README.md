# Praca inżynierska

W niniejszej pracy zbudowano sztuczną sieć neuronową w języku C++, a następnie zaimplementowano wersję programu w architekturze CUDA. Celem było stworzenie wydajnej i poprawnie działającej sieci neuronowej bez wykorzystania gotowych bibliotek do klasyfikatora przypadków w eksperymencie LHCb. Na danych z symulacji Monte Carlo dokonano analizy działania sieci neuronowej w zależności od zastosowanych rozwiązań.


## Przed uruchomieniem

Aby skompilować program należy użyć komendy
```
nvcc -std=c++11 main.cu cpu.cu gpu.cu -o program_name
```

### Dane

Aby sieć neuronowa mogła rozwiązać problem należy przygotować dane do uczenia. W folderze "dane" jest przykładowy zbiór danych po normalizacji i podzielonych na dwa pliki. Jeden zawierający przypadki prawdziwe, drugi fałszywe. 

<p align="center">
  <img src="https://github.com/PiotrWNowak/Deep-Neural-Network-LHCb-/raw/master/analysys/images/1.png">
</p>

### Trening

Zaimplementowane jest do wyboru 5 algorytmów optymalizacji metody gradientu prostego <br />
1 - SGD (Stochastic gradient descent) <br />
2 - Momentum <br />
3 - AdaGrad <br />
4 - RMSprop <br />
5 - Adam <br />

Porównanie kosztu (loss) w epoce treningu sieci dla różnych metod
<p align="center">
  <img src="https://github.com/PiotrWNowak/Deep-Neural-Network-LHCb-/raw/master/analysys/images/2.png">
</p>

Dla 10^5 przypadków w zbiorze treningowym i 10^4 w zbiorze testowym dokonano sprawdzenia
umiejętności rozwiązania problemu klasyfikacji dla sieci o trzech warstwach ukrytych zawierających po 50 neuronów. Użyto algorytmu Adam do uczenia, porównano koszty (loss) dla zbioru treningowego i testowego.
<p align="center">
  <img src="https://github.com/PiotrWNowak/Deep-Neural-Network-LHCb-/raw/master/analysys/images/3.png">
</p>

Dodatkowo sprawdzono wpływ ustawienia progu (Threshold) na dokładność oraz precyzje klasyfikacji.
<p align="center">
  <img src="https://github.com/PiotrWNowak/Deep-Neural-Network-LHCb-/raw/master/analysys/images/4.png">
</p>

### Głębsza analiza oraz opis działania znajdują się w pliku Bachelor thesis.pdf w folderze analysys.

## Authors

**Piotr Nowak** - *Trenowanie sieci neuronowej do klasyfikacji śladów w eksprymencie LHCb przy wykorzystaniu kart graficznych* 
