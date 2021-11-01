#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

#define ALPHA 0.01


void logistic(matrix* x) {
    matrix m = *x;
    for (int i = 0; i < m.rows*m.cols; i++) {
        m.data[i] = 1 / (1 + exp(-1 * m.data[i]));
    }
}

void relu(matrix* x) {
    matrix m = *x;
    for (int i = 0; i < m.rows * m.cols; i++) {
        if (m.data[i] <= 0) {
            m.data[i] = 0;
        }
    }
}

void lrelu(matrix* x) {
    matrix m = *x;
    for (int i = 0; i < m.rows*m.cols; i++) {
        if (m.data[i] <= 0) {
            m.data[i] *= ALPHA;
        }
    }
}

void softmax(matrix* x) {
    matrix m = *x;
    for (int i = 0; i < m.rows; i++) {
        float row_sum = 0;
        for (int j = 0; j < m.cols; j++) {
            float curr_exp = exp(m.data[(i * m.cols) + j]);
            m.data[(i * m.cols) + j] = curr_exp;
            row_sum += curr_exp;
        }
        for (int j = 0; j < m.cols; j++) {
            m.data[(i * m.cols) + j] /= row_sum;
        }
    } 
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row
    switch (a)
    {
        case LOGISTIC: logistic(&y); break;
        case RELU: relu(&y); break;
        case LRELU: lrelu(&y); break;
        case SOFTMAX: softmax(&y); break;
        default: break;
    }
    return y;
}

float ddx_logistic(float x) {
    float logistic_x = 1 / (1 + exp(-1*x));
    return (logistic_x) * (1 - logistic_x);
}

float ddx_relu(float x) {
    return (x > 0 ? 1 : 0);
}

float ddx_lrelu(float x) {
    return (x > 0 ? 1 : ALPHA);
}

float ddx_softmax(float x) {
    return 1.0;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1
    float (*f_dash)(float);
    switch (a)
    {
        case LOGISTIC: f_dash = ddx_logistic; break;
        case RELU: f_dash = ddx_relu; break;
        case LRELU: f_dash = ddx_lrelu; break;
        case SOFTMAX: f_dash = ddx_softmax; break;
        default: break;
    }
    for (int i = 0; i < dx.cols*dx.rows; i++) {
        dx.data[i] = f_dash(x.data[i]) * dy.data[i];
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
