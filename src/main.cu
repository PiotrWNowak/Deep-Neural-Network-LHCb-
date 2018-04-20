#include "nn.h"

int main(){
  Neural_network a(11,1,1000);
  a.set_data_size(1000);
  a.import_data("../data/data1.txt", 0, a.get_data_size()/2);
  a.import_data("../data/data0.txt", a.get_data_size()/2, a.get_data_size());
  a.set_data(900,100);
  a.shuffle();
  a.set_hiperparameters(100, adam, 0.01, 0.1);
  a.use_GPU();
  a.train_with_GPU(20);

}
