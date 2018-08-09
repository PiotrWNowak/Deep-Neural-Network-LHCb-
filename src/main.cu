#include "nn.h"

int main(){
  Neural_network a(11,2,20);
  a.set_data_size(10000);
  a.import_data("../data/data1.txt", 0, a.get_data_size()/2);
  a.import_data("../data/data0.txt", a.get_data_size()/2, a.get_data_size());
  a.set_data(9000,1000);
  a.shuffle();
  a.set_hiperparameters(10, adam, 0.0001, 0.1);
  a.use_GPU();
  a.train(10);
  a.save_model("weights.txt");
}
