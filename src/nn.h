#ifndef nn_h
#define nn_h

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <cstring>
#include <memory>
#include <sys/time.h>
#include <math.h>
#include <cmath>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

enum gradient_type{
  normal_gradient = 1,
  momentum = 2,
  adagrad = 3,
  RMSprop = 4,
  adam = 5
};

class Neural_network{
  private:
    const int input;
    const int output = 2;
    const int layers_number;
    int data_size;
    int training_size;
    int test_size;
    int batch_size;
    int gradient;
    double learning_rate;
    double error;
    double loss;
    bool GPU_bool;
    int* layers_size;
    double* data_X;
    double* data_Y;
    double* training_X;
    double* training_Y;
    double* test_X;
    double* test_Y;
    double** w;
    double** w_gradient;
    double** w_gradient_old;
    double** w_gradient_old2;
    double** l;
    double** a_l;
    double** d_l;
    double** delta;

    dim3 block;
  	dim3 block2;
  	dim3 grid;
    double* training_X_GPU;
    double* training_Y_GPU;
    double* test_X_GPU;
    double* test_Y_GPU;
    double** w_GPU;
    double* error_GPU;
    double* loss_GPU;
    double* error_CPU;
    double* loss_CPU;
  public:
    Neural_network(int, int, int);
    void set_data_size(int);
    int get_data_size();
    void import_data(std::string, int, int);
    void shuffle();
    void set_data(int, int);
    void set_wage_zero(double **);
    void set_hiperparameters(int, gradient_type, double, double);
    void use_CPU();
    void train(int);
    void train_with_CPU(int);
    void feed_forward(double*, double*, double*, double*);
    void matrix_multiplication(int);
    void matrix_activation(int);
    void softmax();
    void error_check(double*, double*, double*);
    void error_calculate(int);
    void gradient_calculate(int);
    void update();
    void normal_gradient_update(int);
    void momentum_update(int);
    void adagrad_update(int);
    void RMSprop_update(int);
    void adam_update(int);
    void wage_max_min(double, double);

    void use_GPU();
    void train_with_GPU(int);
    void feed_forward_GPU(double*, double*, double*, double*);
    void update_GPU();
};

double sigmoid(double);
double d_sigmoid(double);
double relu(double);
double d_relu(double);
double lrelu(double);
double d_lrelu(double);

__global__ void matrix_multiplication_GPU(double *l2, double *l1, double *w, int l2_size, int l1_size, int batch_size);
__global__ void matrix_activation_GPU( double *l, double *a_l, double *d_l, int l_size, int batch_size);
__global__ void softmax_GPU( double *l, double *a_l, int l_size, int batch_size);
__global__ void error_check_GPU(double *Y, double *a_l, double *delta, double *d_l, double *error, double *loss, int output, int batch_size);
__global__ void error_calculate_GPU(double *l2, double *l1, double *w, int l2_size, int l1_size, int batch_size);
__global__ void set_GPU(double *w, int l1_size, int l2_size, double d);
__global__ void gradient_calculate_GPU(double *a_l1, double *w, double *delta, double *d_l2, int l1_size, int l2_size, int batch_size);
__global__ void normal_gradient_update_GPU(double *w, double *w_g, int l1_size, int l2_size, double learning_rate);
__global__ void momentum_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate);
__global__ void adagrad_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate);
__global__ void RMSprop_update_GPU(double *w, double *w_g, double *w_g_old, int l1_size, int l2_size, double learning_rate);
__global__ void adam_update_GPU(double *w, double *w_g, double *w_g_old, double *w_g_old2, int l1_size, int l2_size, double learning_rate);


#endif
