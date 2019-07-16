import numpy as np
import utils

def get_mask(conv):
    
    """
    args: convolution layer
    description: used for the backprop approximation in maxpool gradient. 
                 Returns the maximum values of conv (passed into pool during 
                 the forward operation). Assigns true to max, and false to all other.
    returns: mask of max values from convolution layer
    """

    mask = conv == np.max(conv)
    return mask

def maxpool_gradient(self, conv_prev, dpool, filSize, stride):
    
    filHeight, filWidth = filSize
    
    """
    arguments:  conv_prev - input into current pooling layer (normally convolution) (prev. depth, prev. height, prev. width)
                dpool - gradient of the loss w.r.t following pooling layer (curr. depth, height, width)
                filSize - filter size of pooling operation (m, m)
                stride - stride size of pooling operation (int)
    description: loops through the dimensions of the pooling layer, applies an (m, m) filter with some stride, and 
                 returns an approximation of the gradient (returning back the max value).
    returns:  gradient of loss w.r.t conv_prev
    """
    
    d_prev, h_prev, w_prev  = conv_prev.shape     # shape of input into pool layer
    d, h, w = dpool.shape                         # shape of pool layer
    
    # initialize derivative
    dconv_prev = np.zeros(conv_prev.shape)       
    
    for i in range(d):
        height = 0
        for j in range(w):
            width = 0
            for k in range(h):
                conv_prev_slice = conv_prev[i, height:height+filHeight, width:width+filWidth]
                mask = get_mask(conv_prev_slice)
                dconv_prev[i, height:height+filHeight, width:width+filWidth] += np.multiply(mask, dpool[i, j, k])
                
                width += stride
            height += stride
    return dconv_prev

def convolution_gradient(self, dout, prev_input, W, b, filSize, stride):
    
    """
    args:  dout - gradient of loss w.r.t convolution output (curr. depth, curr. height, curr. width)
           prev_input - input into convolution layer (usually a pooling layer) (prev. depth, prev. height, prev. width)
           W - kernel of convolution layer (curr. depth, prev. depth, height, width)
           b - bias of convolution layer (curr. depth, 1)
           filSize - filter size of pooling operation (m, m)
           stride - stride size of pooling operation (int)
    description:  loops through the current convolution layer to obtain three gradients (mentioned below)
    returns: dprev_input - gradient of cost with respect to conv_prev
             dW - gradient of loss w.r.t convolution kernel 
             db - gradient of loss w.r.t convolution bias 
    """
    
    (d_prev, h_prev, w_prev) = prev_input.shape       # input shape to conv layer
    (_, _, hKernel, wKernel) = W.shape                # kernel shape of conv layer 
    (d, h, w) = dout.shape                            # gradient shape of conv layer derivative
    
    # initialize derivatives
    dprev_input = np.zeros(prev_input.shape)                      
    dW = np.zeros(W.shape) 
    db = np.zeros(b.shape)

    for j in range(d):
        height = 0
        for k in range(h):
            width = 0
            for l in range(w):
                prev_slice = prev_input[:, height:height+hKernel, width:width+wKernel]
                dW[j] += dout[j, k, l] * prev_slice
                dprev_input[:, height:height+hKernel, width:width+wKernel] += dout[j,k,l] * W[j] 
        
        db[j] += np.sum(dout[j])

    return dprev_input, dW, db


def fc_grad_second(self, label, output, prev_input):
    
    """
    arguments: label - true value (10, 1)
               output - NN prediction (10, 1)
               prev_input - previous fc layer (64, 1)
    description: calculates the gradient w.r.t the last fc layer
    returns: gradf2 - gradient of loss w.r.t the output (10, 1)
             gradw4 - gradient of loss w.r.t weight (10, 64)
             gradb4 - gradient of loss w.r.t bias (10, 1)
    """
    
    gradf2 = output - label
    gradw4 = gradf2.dot(prev_input.T)
    gradb4 = np.sum(gradf2, axis = 0)
    return(gradf2,gradw4,gradb4)

def fc_grad_first(self, gradfc2, gradweight, prev_input, curr_input):
    
    """
    arguments: gradfc2 - gradient of loss w.r.t fc2 (10, 1)
               gradweight - gradient of loss w.r.t fc2 weight (10, 64)
               prev_input - previous pooling layer (1, 1600)
               curr_input - input into fc layer 1 (not activated) (64, 1)
    description: calculates the gradient w.r.t the first fc layer
    returns: gradf1 - gradient of loss w.r.t the output (64, 1)
             gradw3 - gradient of loss w.r.t weight (64, 1600)
             gradb3 - gradient of loss w.r.t bias (64, 1)
    """
        
    gradf1 = gradweight.T.dot(gradfc2) 
    gradw3 = gradf1.dot(prev_input)
    gradb3 = np.sum(gradf1,axis = 0)
    return(gradf1, gradw3, gradb3)