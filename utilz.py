import numpy as np

def get_output(self, image, kernel_dim, stride, num_filters):
    """
    initiate output
    """
    numImages, heightImages, widthImages, depthImages = image.shape
    output_volume = (widthImages-kernel_dim)/stride + 1
    if output_volume.is_integer() == False:
        raise NotImplementedError
    else:
        output_volume = int(output_volume)
        return np.zeros((numImages,output_volume,output_volume,num_filters))

def get_parameters(self, image, kernel_dim):
    """
    initiate weights and biases
    """
    depth = image.shape[-1]
    weight = np.random.randn(image.shape[0],kernel_dim, kernel_dim, depth)
    bias = np.random.randn(1)
    return (weight,bias)

def get_forward_dict(self):
    X,h1,h2 = self.forward_dict['image'],self.forward_dict['conv1'],self.forward_dict['conv2']
    p1,p2 = self.forward_dict['pool1'],self.forward_dict['pool2']
    return (X,h1,p1,h2,p2)

def get_parameter_dict(self):
    k1,k2,w1,w2 = self.weight_dict['conv1'][0],self.weight_dict['conv2'][0],self.weight_dict['f1'],self.weight_dict['f2']
    b1,b2,b1_f,b2_f = self.weight_dict['conv1'],self.weight_dict['conv2'],self.weight_dict['f1'],self.weight_dict['f2']
    return (k1,b1,k2,b2,w1,b1_f,w2,b2_f)

def relu(self,image):
    return np.maximum(0,image)

def softmax(self, image):
    return np.exp(image)/np.sum(np.exp(image))