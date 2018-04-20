#include "nn.h"

Neural_network::Neural_network( int m_input, int m_hidden_layers, int m_layers_size) : input(++m_input), layers_number(m_hidden_layers+2){
  layers_size= new int[this->layers_number];
  layers_size[0] = input;
  for(int i=1; i<layers_number-1; i++) layers_size[i] = m_layers_size;
  layers_size[layers_number-1] = output;

  //srand( time( NULL ) );
  srand(10);

  w = new double* [layers_number-1];
  w_gradient = new double* [layers_number-1];
  w_gradient_old = new double* [layers_number-1];
  w_gradient_old2 = new double* [layers_number-1];
  for(int k=0; k<layers_number-1; k++){
    w[k]=new double[layers_size[k]*layers_size[k+1]];
  }

  l = new double* [layers_number-1];
  d_l = new double* [layers_number-1];
  delta = new double* [layers_number-1];
  a_l = new double* [layers_number];
}


void Neural_network::set_data_size(int data_size){
  this->data_size=data_size;
  data_X = new double[data_size*input];
  data_Y = new double[data_size];
}

int Neural_network::get_data_size(){
  return this->data_size;
}

void Neural_network::import_data(std::string s, int from, int to){
  std::fstream plik( s, std::ios::in );
  if( plik.good() ){
    for(int i=from;i<to;i++){
      plik >> data_Y[i];
      for (int j=0;j<input-1;j++){
        plik >> data_X[i*input+j];
      }
      data_X[i*input+input-1]=1;
    }
  }
  plik.close();
}

void Neural_network::shuffle(){
  int i,j;
  double a;
  for(int k=0; k<data_size/2;k++){
    i=(int)rand() % data_size;
    j=(int)rand() % data_size;

    for(int l=0; l<input; l++){
      a=data_X[i*input+l];
      data_X[i*input+l]=data_X[j*input+l];
      data_X[j*input+l]=a;
  }
    a=data_Y[i];
    data_Y[i]=data_Y[j];
    data_Y[j]=a;
  }
  for(int i=0;i<training_size;i++){
    training_Y[i]=data_Y[i];
    for (int j=0;j<input;j++){
      training_X[i*input+j]=data_X[i*input+j];
    }
  }
  for(int i=0;i<test_size;i++){
    test_Y[i]=data_Y[(i+training_size)];
    for (int j=0;j<input;j++){
      test_X[i*input+j]=data_X[(i+training_size)*input+j];
    }
  }
}

void Neural_network::set_data(int training_size, int test_size){
  this->training_size=training_size;
  this->test_size=test_size;

  training_X = new double[training_size*input];
  training_Y = new double[training_size];
  test_X = new double[test_size*input];
  test_Y = new double[test_size];
}

void Neural_network::set_wage_zero(double **w){
  for(int k=0; k<layers_number-1; k++){
    for( int i=0;i<layers_size[k];i++){
      for( int j=0;j<layers_size[k+1];j++){
        w[k][i*layers_size[k+1]+j]=0;
      }
    }
  }
}

void Neural_network::set_hiperparameters(int batch_size, gradient_type gradient, double initiation_skale, double learning_rate){
  this->batch_size = batch_size;
  this->gradient = gradient;
  this->learning_rate=learning_rate;

  for(int k=0; k<layers_number-1; k++){
    for( int i=0;i<layers_size[k];i++){
      for( int j=0;j<layers_size[k+1];j++){
        w[k][i*layers_size[k+1]+j]=((double)rand()/RAND_MAX)*initiation_skale;
      }
    }
  }
}

void Neural_network::use_CPU(){
  for(int k=0; k<layers_number-1; k++){
    w_gradient[k]=new double[layers_size[k]*layers_size[k+1]];
    w_gradient_old[k]=new double[layers_size[k]*layers_size[k+1]];
    w_gradient_old2[k]=new double[layers_size[k]*layers_size[k+1]];
  }

  set_wage_zero(w_gradient);
  set_wage_zero(w_gradient_old);
  set_wage_zero(w_gradient_old2);

  for(int i=0; i<layers_number-1; i++){
    l[i]=new double[layers_size[i+1]*batch_size];
    d_l[i]=new double[layers_size[i+1]*batch_size];
    delta[i]=new double[layers_size[i+1]*batch_size];
    a_l[i+1]=new double[layers_size[i+1]*batch_size];
  }
}

void Neural_network::train(int epoch_number){
  for(int epoch=0; epoch<epoch_number; epoch++){
    error=loss=0;
    for(int batch=0; batch<training_size; batch+=batch_size){
      feed_forward( &training_X[batch*input],&training_Y[batch], &error, &loss);
      for(int i=layers_number-2; i>0; i--){
        error_calculate(i);
      }
      set_wage_zero(w_gradient);
      for(int i=0; i<layers_number-1; i++){
        gradient_calculate(i);
      }
      update();
    }
    std::cout<<"Epoch "<<epoch<<" Training loss = "<<loss/training_size*batch_size<<" error = "<<1-(error/training_size*batch_size);
    error=loss=0;
    for(int batch=0; batch<test_size; batch+=batch_size){
      feed_forward( &test_X[batch*input],&test_Y[batch], &error, &loss);
    }
    std::cout<<"  Validation loss "<<loss/test_size*batch_size<<" error = "<<1-(error/test_size*batch_size)<<std::endl;
  }
}

void Neural_network::feed_forward(double* X, double* Y, double* error, double* loss){
  a_l[0] = X;
  for(int i=0; i<layers_number-2; i++){
    matrix_multiplication(i);
    matrix_activation(i);
  }
  matrix_multiplication(layers_number-2);
  softmax();
  error_check(Y, error, loss);
}

void Neural_network::matrix_multiplication(int n){
  for(int j=0;j<batch_size;j++){
    for(int i=0;i<layers_size[n+1];i++){
      l[n][j*layers_size[n+1]+i]=0;
      for(int k=0;k<layers_size[n];k++) l[n][j*layers_size[n+1]+i]+=a_l[n][j*layers_size[n]+k]*w[n][k*layers_size[n+1]+i];
    }
  }
}

void Neural_network::matrix_activation(int n){
  for(int j=0;j<batch_size;j++){
    for(int i=0;i<layers_size[n+1];i++){
      a_l[n+1][j*layers_size[n+1]+i]=lrelu(l[n][j*layers_size[n+1]+i]);
      d_l[n][j*layers_size[n+1]+i]=d_lrelu(a_l[n+1][j*layers_size[n+1]+i]);
    }
  }
}

void Neural_network::softmax(){
  for(int j=0;j<batch_size;j++){
    double sum=0;
    for(int i=0;i<output;i++){
      a_l[layers_number-1][j*output+i]=exp(l[layers_number-2][j*output+i]);
      sum+=a_l[layers_number-1][j*output+i];
    }
    for(int i=0;i<output;i++) a_l[layers_number-1][j*output+i]/=sum;
  }
}

void Neural_network::error_check(double* Y, double* m_error, double* m_loss){
  int y;
  double loss=0, error=0;
	for(int j=0;j<batch_size;j++){
		for(int i=0;i<output;i++){
      d_l[layers_number-2][j*output+i]=1.;
			y=(Y[j]-i)*(Y[j]-i);
			delta[layers_number-2][j*output+i]=a_l[layers_number-1][j*output+i]-y;
			loss-=y*log(a_l[layers_number-1][j*output+i]);
		}
		int wynik;
		if(a_l[layers_number-1][j*output]>a_l[layers_number-1][j*output+1]) wynik=1;
		else wynik=0;
		if(wynik==Y[j]) error++;
	}
	(*m_loss)+=loss/batch_size;
	(*m_error)+=error/batch_size;
}

void Neural_network::error_calculate(int n){
  for(int j=0;j<batch_size;j++){
    for(int i=0;i<layers_size[n];i++){
      delta[n-1][j*layers_size[n]+i]=0;
      for(int k=0;k<layers_size[n+1];k++) delta[n-1][j*layers_size[n]+i]+=delta[n][j*layers_size[n+1]+k]*w[n][i*layers_size[n+1]+k];
    }
  }
}

void Neural_network::gradient_calculate(int n){
  for(int i=0;i<layers_size[n];i++){
		for(int k=0;k<layers_size[n+1];k++){
			double w_update=0;
			for(int j=0;j<batch_size;j++){
        //double update=a_l1[j*l1_size+i]*delta[j*l2_size+k]*d_l2[j*l2_size+k];
        //if(isnan(update)==0) w_update+=update;
        w_update+=a_l[n][j*layers_size[n]+i]*delta[n][j*layers_size[n+1]+k]*d_l[n][j*layers_size[n+1]+k];
      }
			w_update/=batch_size;
			w_gradient[n][i*layers_size[n+1]+k]=w_update;
		}
	}
}

void Neural_network::update(){
  switch (gradient){
    case 1:
      for(int i=0; i<layers_number-1; i++) normal_gradient_update(i);
      break;
    case 2:
      for(int i=0; i<layers_number-1; i++) momentum_update(i);
      break;
    case 3:
      for(int i=0; i<layers_number-1; i++) adagrad_update(i);
      break;
    case 4:
      for(int i=0; i<layers_number-1; i++) RMSprop_update(i);
      break;
    case 5:
      for(int i=0; i<layers_number-1; i++) adam_update(i);
      break;
  }
}

void Neural_network::normal_gradient_update(int n){
  for(int i=0;i<layers_size[n];i++){
    for(int k=0;k<layers_size[n+1];k++){
      w[n][i*layers_size[n+1]+k]-=w_gradient[n][i*layers_size[n+1]+k]*learning_rate;
    }
  }
}

void Neural_network::momentum_update(int n){
  for(int i=0;i<layers_size[n];i++){
		for(int k=0;k<layers_size[n+1];k++){
      double v=0;
      v=0.8*w_gradient_old[n][i*layers_size[n+1]+k]+learning_rate*w_gradient[n][i*layers_size[n+1]+k];
      if(v>1) v=1;
      if(v<-1) v=-1;
			w[n][i*layers_size[n+1]+k]-=v;
      w_gradient_old[n][i*layers_size[n+1]+k]=v;
		}
	}
}


void Neural_network::adagrad_update(int n){
  for(int i=0;i<layers_size[n];i++){
		for(int k=0;k<layers_size[n+1];k++){
      w_gradient_old[n][i*layers_size[n+1]+k]+=w_gradient[n][i*layers_size[n+1]+k]*w_gradient[n][i*layers_size[n+1]+k];
      double v=0;
      v=learning_rate*w_gradient[n][i*layers_size[n+1]+k]/(sqrt(w_gradient_old[n][i*layers_size[n+1]+k]+pow(1,-8)));
      //if(v>1) v=1;
      //if(v<-1) v=-1;
			w[n][i*layers_size[n+1]+k]-=v;
		}
	}
}

void Neural_network::RMSprop_update(int n){
  for(int i=0;i<layers_size[n];i++){
		for(int k=0;k<layers_size[n+1];k++){
      w_gradient_old[n][i*layers_size[n+1]+k]=0.1*(w_gradient[n][i*layers_size[n+1]+k]*w_gradient[n][i*layers_size[n+1]+k])+0.9*w_gradient_old[n][i*layers_size[n+1]+k];
			w[n][i*layers_size[n+1]+k]-=w_gradient[n][i*layers_size[n+1]+k]*learning_rate/(sqrt(w_gradient_old[n][i*layers_size[n+1]+k]+pow(1,-8)));
		}
	}
}

void Neural_network::adam_update(int n){
  double B1=0.9;
  double B2=0.999;
  double m;
  double v;
	for(int i=0;i<layers_size[n];i++){
		for(int k=0;k<layers_size[n+1];k++){
      w_gradient_old[n][i*layers_size[n+1]+k]=B1*w_gradient_old[n][i*layers_size[n+1]+k]+(1-B1)*w_gradient[n][i*layers_size[n+1]+k];
      w_gradient_old2[n][i*layers_size[n+1]+k]=B2*w_gradient_old2[n][i*layers_size[n+1]+k]+(1-B2)*w_gradient[n][i*layers_size[n+1]+k]*w_gradient[n][i*layers_size[n+1]+k];
      m=w_gradient_old[n][i*layers_size[n+1]+k]/(1-B1);
      v=w_gradient_old2[n][i*layers_size[n+1]+k]/(1-B2);
			w[n][i*layers_size[n+1]+k]-=m*learning_rate/(sqrt(v+pow(1,-8)+0.));
		}
	}
}

void Neural_network::wage_max_min(double maximum, double minimum){
  for(int k=0; k<layers_number-1; k++){
    for( int i=0;i<layers_size[k];i++){
      for( int j=0;j<layers_size[k+1];j++){
        if(w[k][i*layers_size[k+1]+j] > maximum) w[k][i*layers_size[k+1]+j] = maximum;
        if(w[k][i*layers_size[k+1]+j] < minimum) w[k][i*layers_size[k+1]+j] = minimum;
      }
    }
  }
}

double sigmoid(double y){
	return 1./(1.+exp(-y));
}
double d_sigmoid(double y){
	return y*(1.-y);
}

double relu(double y){
	if (y>0) return y;
	else return 0;
}
double d_relu(double y){
	if (y>0) return 1;
	else return 0;
}

double lrelu(double y){
	if (y>0) return y;//min(y,1.);
	else return 0.01*y;
}
double d_lrelu(double y){
	if (y>0) return 1;
	else return 0.01;
}
