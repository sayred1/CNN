import numpy as np

def initConv(self, image, kernel, stride):
    """
    initiate conv output
    """
    numInFil, heightImages, widthImages = image.shape
    numOutFil, _, heightKernel, widthKernel = kernel.shape
    
    outHeight = (widthImages-heightKernel)/stride + 1

    if outHeight.is_integer() == False:
        raise NotImplementedError
    
    outHeight = outWidth = int(outHeight)
    return np.zeros((numOutFil, outHeight,outWidth))
    
def initPool(self, image, filDim, stride):
    """
    initiate pool output
    """
    numInFil, heightImages, widthImages = image.shape
    heightFil, widthFil = filDim
    
    outHeight = (widthImages-heightFil)/stride + 1
    if outHeight.is_integer() == False:
        raise NotImplementedError
    else:
        outHeight = outWidth = int(outHeight)
        return np.zeros((numInFil,outHeight,outWidth))


def relu(self,image):
    return np.maximum(0,image)

def softmax(self, image):
    return np.exp(image)/np.sum(np.exp(image))