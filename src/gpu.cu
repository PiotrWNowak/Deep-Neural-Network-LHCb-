#include "nn.h"

void Neural_network::use_GPU(){

  block = batch_size;
	block2 = layers_size[1];
	grid = layers_size[1];

  cudaMalloc((double**)&training_X_GPU, training_size*input*sizeof(double));
	cudaMalloc((double**)&training_Y_GPU, training_size*sizeof(double));
	cudaMalloc((double**)&test_X_GPU, test_size*input*sizeof(double));
	cudaMalloc((double**)&test_Y_GPU, test_size*sizeof(double));

  w_GPU = new double* [layers_number-1];
	for( int i=0;i<layers_number-1;i++){
		cudaMalloc((double**)&(w_GPU[i]), layers_size[i]*layers_size[i+1]*sizeof(double));
		cudaMalloc((double**)&(w_gradient[i]), layers_size[i]*layers_size[i+1]*sizeof(double));
		cudaMalloc((double**)&(w_gradient_old[i]), layers_size[i]*layers_size[i+1]*sizeof(double));
		cudaMalloc((double**)&(w_gradient_old2[i]), layers_size[i]*layers_size[i+1]*sizeof(double));
		cudaMemcpy(w_GPU[i], w[i], layers_size[i]*layers_size[i+1]*sizeof(double), cudaMemcpyHostToDevice);
    set_wage_zero_GPU<<< grid, block >>>(w_gradient[i], layers_size[i], layers_size[i+1]);
    set_wage_zero_GPU<<< grid, block >>>(w_gradient_old[i], layers_size[i], layers_size[i+1]);
    set_wage_zero_GPU<<< grid, block >>>(w_gradient_old[i], layers_size[i], layers_size[i+1]);
  }

  for(int i=0; i<layers_number-1; i++){
    cudaMalloc((double**)&(l[i]), layers_size[i+1]*batch_size*sizeof(double));
    cudaMalloc((double**)&(d_l[i]), layers_size[i+1]*batch_size*sizeof(double));
    cudaMalloc((double**)&(delta[i]), layers_size[i+1]*batch_size*sizeof(double));
    cudaMalloc((double**)&(a_l[i+1]), layers_size[i+1]*batch_size*sizeof(double));
  }
  cudaMalloc((double**)&error_GPU, batch_size*sizeof(double));
  cudaMalloc((double**)&loss_GPU, batch_size*sizeof(double));
  error_CPU=new double [batch_size];
  loss_CPU=new double [batch_size];
}

void Neural_network::train_with_GPU(int epoch_number){
  cudaMemcpy(training_X_GPU, training_X, training_size*input*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(training_Y_GPU, training_Y, training_size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(test_X_GPU, test_X, test_size*input*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(test_Y_GPU, test_Y, test_size*sizeof(double), cudaMemcpyHostToDevice);
  for(int epoch=0; epoch<epoch_number; epoch++){
    error=loss=0;
    for(int batch=0; batch<training_size; batch+=batch_size){
      feed_forward_GPU( &training_X_GPU[batch*input],&training_Y_GPU[batch], &error, &loss);
      for(int i=layers_number-2; i>0; i--){
        error_calculate_GPU<<< grid, block >>>(delta[i], delta[i-1], w_GPU[i], layers_size[i+1], layers_size[i], batch_size);
      }
      for( int i=0;i<layers_number-1;i++){
        set_wage_zero_GPU<<< grid, block >>>(w_gradient[i], layers_size[i], layers_size[i+1]);
        gradient_calculate_GPU<<< grid, block2 >>>(a_l[i], w_gradient[i], delta[i], d_l[i], layers_size[i], layers_size[i+1], batch_size);
      }
      update_GPU();
    }
    std::cout<<"Epoch "<<epoch<<" Training loss = "<<loss/training_size*batch_size<<" error = "<<1-(error/training_size*batch_size);
    error=loss=0;
    for(int batch=0; batch<test_size; batch+=batch_size){
      feed_forward_GPU( &test_X_GPU[batch*input],&test_Y_GPU[batch], &error, &loss);
    }
    std::cout<<"  Validation loss "<<loss/test_size*batch_size<<" error = "<<1-(error/test_size*batch_size)<<std::endl;
  }
  for( int i=0;i<layers_number-1;i++){
    cudaMemcpy(w[i], w_GPU[i], layers_size[i]*layers_size[i+1]*sizeof(double), cudaMemcpyDeviceToHost);
  }
}

void Neural_network::feed_forward_GPU(double* X, double* Y, double* error, double* loss){
  a_l[0] = X;
  for(int i=0; i<layers_number-2; i++){
    matrix_multiplication_GPU<<< grid, block >>>(l[i], a_l[i], w_GPU[i], layers_size[i+1], layers_size[i], batch_size);
    matrix_activation_GPU<<< grid, block >>>(l[i], a_l[i+1], d_l[i], layers_size[i+1], batch_size);
  }
  matrix_multiplication_GPU<<< grid, block >>>(l[layers_number-2], a_l[layers_number-2], w_GPU[layers_number-2], layers_size[layers_number-1], layers_size[layers_number-2], batch_size);
  softmax_GPU<<< grid, block >>>( l[layers_number-2], a_l[layers_number-1], output, batch_size);
  error_check_GPU<<< grid, block >>>(Y, a_l[layers_number-1], delta[layers_number-2], d_l[layers_number-2], error_GPU, loss_GPU, output, batch_size);
  cudaMemcpy(error_CPU, error_GPU, batch_size*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(loss_CPU, loss_GPU, batch_size*sizeof(double), cudaMemcpyDeviceToHost);
  for(int i=1;i<batch_size;i++) error_CPU[0]+=error_CPU[i];
  for(int i=1;i<batch_size;i++) loss_CPU[0]+=loss_CPU[i];
  (*error)+=error_CPU[0]/batch_size;
  (*loss)+=loss_CPU[0]/batch_size;
}

void Neural_network::update_GPU(){
  switch (gradient){
    case 1:
      for(int i=0; i<layers_number-1; i++)
        normal_gradient_update_GPU<<< grid, block >>>(w_GPU[i], w_gradient[i], layers_size[i], layers_size[i+1], learning_rate);
      break;
    case 2:
      for(int i=0; i<layers_number-1; i++)
        momentum_update_GPU<<< grid, block >>>(w_GPU[i], w_gradient[i], w_gradient_old[i], layers_size[i], layers_size[i+1], learning_rate);
      break;
    case 3:
      for(int i=0; i<layers_number-1; i++)
        adagrad_update_GPU<<< grid, block >>>(w_GPU[i], w_gradient[i], w_gradient_old[i], layers_size[i], layers_size[i+1], learning_rate);
      break;
    case 4:
      for(int i=0; i<layers_number-1; i++)
        RMSprop_update_GPU<<< grid, block >>>(w_GPU[i], w_gradient[i], w_gradient_old[i], layers_size[i], layers_size[i+1], learning_rate);
      break;
    case 5:
      for(int i=0; i<layers_number-1; i++)
        adam_update_GPU<<< grid, block >>>(w_GPU[i], w_gradient[i], w_gradient_old[i], w_gradient_old2[i], layers_size[i], layers_size[i+1], learning_rate);
      break;
  }
}

__global__ void matrix_multiplication_GPU(double *l2, double *l1, double *w, int l2_size, int l1_size, int batch_size){
	unsigned int j = threadIdx.x;
	unsigned int i = blockIdx.x;
		if(i<l2_size){
			l2[j*l2_size+i]=0;
			for(int k=0;k<l1_size;k++) l2[j*l2_size+i]+=l1[j*l1_size+k]*w[k*l2_size+i];
		}
}
__global__ void matrix_activation_GPU( double *l, double *a_l, double *d_l, int l_size, int batch_size){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	for(int i=0; i<l_size*batch_size; i+=gridDim.x*blockDim.x){
		if(i+index<l_size*batch_size){
			if (l[i+index]>0){
				a_l[i+index]=l[i+index];//min(y,1.);
				d_l[i+index]=1;
			}
			else{
				a_l[i+index]=0.01*l[i+index];
				d_l[i+index]=0.01;
			}
		}
	}
}
__global__ void softmax_GPU( double *l, double *a_l, int l_size, int batch_size){
	unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
	if(j<batch_size){
		double sume=0;
		for(int i=0;i<l_size;i++){
			a_l[j*l_size+i]=exp(l[j*l_size+i]);
			sume+=a_l[j*l_size+i];
		}
		for(int i=0;i<l_size;i++) a_l[j*l_size+i]/=sume;
	}
}
__global__ void error_check_GPU(double *Y, double *a_l, double *delta, double *d_l, double *error, double *loss, int output, int batch_size){
	unsigned int j = threadIdx.x + blockIdx.x*blockDim.x;
	if(j<batch_size){
		int y;
		loss[j]=0;
		error[j]=0;
		for(int i=0;i<output;i++){
      d_l[j*output+i]=1.;
			y=(Y[j]-i)*(Y[j]-i);
			delta[j*output+i]=a_l[j*output+i]-y;
			loss[j]-=y*log(a_l[j*output+i]);
		}
		//int wynik;
		if(a_l[j*output]>a_l[j*output+1]) error[j]+=1;
		error[j]+=Y[j];
		if(error[j]==1) error[j]=0;
		else error[j]=1;
	}
}
__global__ void error_calculate_GPU(double *l2, double *l1, double *w, int l2_size, int l1_size, int batch_size){
	unsigned int j = threadIdx.x;
	unsigned int i = blockIdx.x;
	if(i<l1_size){
		l1[j*l1_size+i]=0;
		for(int k=0;k<l2_size;k++) l1[j*l1_size+i]+=l2[j*l2_size+k]*w[i*l2_size+k];
	}
}
__global__ void set_wage_zero_GPU(double *w, int l1_size, int l2_size){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
			w[i+index]=0;
		}
	}
}
__global__ void gradient_calculate_GPU(double *a_l1, double *w, double *delta, double *d_l2, int l1_size, int l2_size, int batch_size){
	unsigned int i = threadIdx.x;
	unsigned int k = blockIdx.x;
	if(i<l1_size){
		if(k<l2_size){
			w[i*l2_size+k]=0;
			for(int j=0;j<batch_size;j++){
        double update=a_l1[j*l1_size+i]*delta[j*l2_size+k]*d_l2[j*l2_size+k];
        if(isnan(update)==0) w[i*l2_size+k]+=update;
        //w[i*l2_size+k]+=a_l1[j*l1_size+i]*delta[j*l2_size+k]*d_l2[j*l2_size+k];
      }
			w[i*l2_size+k]/=batch_size;
		}
	}
}
__global__ void normal_gradient_update_GPU(double *w, double *w_g, int l1_size, int l2_size, double learning_rate){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
			w[i+index]-=w_g[i+index]*learning_rate;
      //if(w[i+index]>2) w[i+index]=2.;
      //if(w[i+index]<-2) w[i+index]=-2.;
		}
	}
}
__global__ void momentum_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
      double v=0;
      v=0.8*w_g_old[i+index]+learning_rate*w_g[i+index];
      //if(v>1) v=1;
      //if(v<-1) v=-1;
			w[i+index]-=v;
      //if(w[i+index]>2) w[i+index]=2.;
      //if(w[i+index]<-2) w[i+index]=-2.;
      w_g_old[i+index]=v;
		}
	}
}
__global__ void adagrad_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
      w_g_old[i+index]+=w_g[i+index]*w_g[i+index];
      double v=0;
      v=learning_rate*w_g[i+index]/(sqrt(w_g_old[i+index]+0.00000001));
      //if(v>1) v=1;
      //if(v<-1) v=-1;
			w[i+index]-=v;
      //if(w[i+index]>2) w[i+index]=2.;
      //if(w[i+index]<-2) w[i+index]=-2.;
		}
	}
}
__global__ void RMSprop_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
      w_g_old[i+index]=0.1*(w_g[i+index]*w_g[i+index])+0.9*w_g_old[i+index];
			w[i+index]-=w_g[i+index]*learning_rate/(sqrt(w_g_old[i+index]+0.00000001));
      //if(w[i+index]>2) w[i+index]=2.;
      //if(w[i+index]<-2) w[i+index]=-2.;
		}
	}
}
__global__ void adam_update_GPU(double *w, double *w_g, double *w_g_old, double *w_g_old2, int l1_size, int l2_size, double learning_rate){
  double B1=0.9;
  double B2=0.999;
  double m;
  double v;
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	int n=l1_size*l2_size;
	for(int i=0; i<n; i+=gridDim.x*blockDim.x){
		if(i+index<n){
      w_g_old[i+index]=B1*w_g_old[i+index]+(1-B1)*w_g[i+index];
      w_g_old2[i+index]=B2*w_g_old2[i+index]+(1-B2)*w_g[i+index]*w_g[i+index];
      m=w_g_old[i+index]/(1-B1);
      v=w_g_old2[i+index]/(1-B2);
			w[i+index]-=m*learning_rate/(sqrt(v+0.00000001+0.));
      if(w[i+index]>2) w[i+index]=2.;
      if(w[i+index]<-2) w[i+index]=-2.;
		}
	}
}
