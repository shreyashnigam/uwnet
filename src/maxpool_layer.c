#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    int pool_offset_multiplier = outw * outh;
    int img_offset_multiplier = l.width * l.height;

    int padding = (l.size - 1)/2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    // From Ed - Each row is a separate image
    for (int img = 0; img < in.rows; img++) {
        float* curr_img = in.data + img*in.cols;
        float* pool_out = out.data + img*out.cols;

        for (int i = 0; i < l.channels; i++) {
            int pool_offset = pool_offset_multiplier * i;
            int img_offset = img_offset_multiplier * i;
            int submatrix_index =0;
            for (int j = 0; j < l.height; j+=l.stride) {
                for (int k = 0; k < l.width; k+=l.stride) {
                    float max = FLT_MIN;
                    for (int m = 0; m < l.size; m++) {
                        for (int n = 0; n < l.size; n++) {
                            int height = j + m - padding;
                            int width = k + n - padding;
                            if (height >= 0 && height < l.height*l.channels && width >= 0 && width < l.width) {
                                int index = width + l.width * height + img_offset;
                                max = MAX(curr_img[index], max);
                            }
                        }
                    }
                    pool_out[pool_offset + submatrix_index] = max;
                    submatrix_index++;
                }
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    int delta_offset_multiplier = outw * outh;
    int img_offset_multiplier = l.width * l.height;

    int padding = (l.size - 1)/2;

    for (int img = 0; img < in.rows; img++) {
        float* layer_img = in.data + img*in.cols;
        float* dy_img = dy.data + img*dy.cols;
        float* dx_img = dx.data + img*dx.cols;

        for (int i = 0; i < l.channels; i++) {
            int delta_offset = delta_offset_multiplier * i;
            int img_offset = img_offset_multiplier * i;
            int submatrix_index =0;
            for (int j = 0; j < l.height; j+=l.stride) {
                for (int k = 0; k < l.width; k+=l.stride) {
                    float max = FLT_MIN;
                    int row = 0;
                    int col = 0;
                    for (int m = 0; m < l.size; m++) {
                        for (int n = 0; n < l.size; n++) {
                            int height = j + m - padding;
                            int width = k + n - padding;
                            if (height >= 0 && height < l.height*l.channels && width >= 0 && width < l.width) {
                                int index = width + l.width * height + img_offset;
                                if (layer_img[index] > max) {
                                    max = layer_img[index];
                                    row = height;
                                    col = width;
                                }
                            }
                        }
                    }
                    dx_img[row * l.width + col + img_offset] += dy_img[delta_offset + submatrix_index];
                    submatrix_index++;
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;

}

