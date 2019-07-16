## To run the CNN, clone CNN with Numpy.ipynb, backward.py, forward.py, and utils.py.

#### As of now, the number of layers are held static, as are the dimensions of parameters. CNN with Numpy.ipynb is composed of the network layout, while backward.py, forward.py, and utils.py hold functions used in the model.

#### The current setup is composed of input --> conv + relu --> pool --> conv + relu --> pool --> flatten --> fc1 -- fc2 (prediction). Since this is set, the parameters are already initialized so as of right now, simpy set a batch size (number of images from the dataset) and a number of epochs (how many times to run the batch). 
