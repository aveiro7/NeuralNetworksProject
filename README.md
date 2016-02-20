Neural Networks Project
==================

The main goal
-----------------

The main goal of the project was to obtain 75% accuracy on the test batch to solve the classification problem on the CIFAR10 set. 

The input
-----------
The input data were 32x32 images, each containing exactly one visible element that without ambiguity could be assigned to one out of ten classes. 

Other restrictions
---------------------
The required accuracy should be achieved using a deep neural network with convolutions (or other, alternative method, such as SVM). The more tricks involved, the better. The recommended tool to use within the project was Theano.


Tools used
-------------
My project was implemented using Lasagne, a lightweight library based on Theano. The main advantage of such approach is the simplicity and ease when creating a network. With this way of network creation, the programmer does not need to bother about the backpropagation algorithm, as it is implemented as a integral part of Theano.


Accuracy achieved
-----------------------
The network implemented for this project achieved about 76.1% accuracy. Perhaps the score could be improved with longer training phase (more epochs), but this result was satisfactory for the author.


Network architecture
-------------------------
The input layer was basically a 32x32 pixel image. As usual for the classification problems, the output layer was the softmax layer (fully connected), with 10 neurons indicating each possible class. Between those two layers there were 5 hidden layers: 
* convolutional layer with 128 filters, analysing 5x5 image subparts, 
* pooling layer, taking 2x2 subparts of the previous layer, 
* again a convolutional layer, this time with 32 filters and analysing 5x5 subparts of the previous layer, 
* again a pooling layer that takes 2x2 subparts from the previous layer 
* and finally the fully connected layer with 256 neurons.

For the network to learn better there is dropout added in the two last layers, with dropout rate equal to 0.5.


Implemented mechanisms
-------------------------------

As stated above, Theano has the backpropagation algorithm implemented as an integral part of itself. The loss function that is used is the categorical crossentropy (implemented through Lasagne). The parameters (weights and biases) are updated using the momentum technique, which also is implemented through Lasagne. The learning rate (at the beginning equal to 0.01) is descending every 20 epochs, each time by factor of 2. The data is processed in small (100 elements each) minibatches -- every epoch of training iterates through randomly partitoned minibatches. The partition of validation and test data is deterministic.


Possible improvements
----------------------------

The number of epochs of training is fixed and set to 175 (value found experimentally) -- perhaps the better solution would be to use early stopping to prevent from overfitting and find the best point to end the computation. The other direction in which to look for improvement would be increasing the number of the hidden layers (perhaps some more convolutional layers) and maybe number of filters for each (convolutional) layer. 

Maybe some kind of input preprocessing would also improve the network’s performance. In some cases the decorrelation of the provided data leads to major successes -- the use of PCA could possibly improve the result.
