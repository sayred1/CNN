import utils
import numpy as np

def convolve(self, image, kernel, bias, stride):
    
    """
    arguments: image - input into convolution layer (X or pool) (curr. depth, height, width, prev. depth)
               kernel - convolutional kernel to perform feature mapping (curr. depth, pre. depth, height, width)
               bias - (curr.depth, 1)
               stride - number of slides in the x and y directions of the array per filter (int)
    description: performs convolution over X or pooling inputs
    returns: convolve (depth, height, width)
    """

    # initialize conv
    conv = utils.initConv(self, image, kernel, stride)    
    
    # obtain shape of image and kernel
    filtInput, heightInput, widthInput = image.shape
    filtK, _, heightK, widthK = kernel.shape
    

    for i in range(filtK):
        height = 0
        for j in range(heightK): 
            width = 0
            for k in range(widthK):
                imgSlice = image[:,height:height+heightK,width:width+widthK]
                conv[i,j,k] = np.sum(kernel[i]*imgSlice) + bias[i]
                width += stride
            height += stride

    return conv

def pool(self, conv, filSize, stride):
    
    """
    arguments: conv - convolution layer (depth, prev. depth, height, width)
               filSize - size of pooling filter (height, width)
               stride - number of slides in the x and y directions of the array per filter (int)
    description: performs maxpooling over convolution inputs
    returns: maxpool (depth, height, width)
    """
    
    depthConv, heightConv, widthConv = conv.shape
    filHeight, filWidth = filSize
    
    pool = utils.initPool(self, conv, filSize, stride)
    _, heightImages,widthImages = pool.shape

    for i in range(depthConv):
        height = 0
        for j in range(heightImages):
            width = 0
            for k in range(widthImages):
                temp = conv[i,height:height+filHeight,width:width+filWidth]
                pool[i,j,k] = np.max(temp)                    
                width+=stride
            height+=stride
    return pool

def loss(self, output, label): 
    """
    arguments: output - CNN prediction (1, 10)
               label - true value (1, 10)
    description: returns the loss of the neural network per example
    returns: loss
    """
    
    loss = -np.sum(label * np.log(output))
    return loss
