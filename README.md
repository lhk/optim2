# optim2
A numpy based implementation of backpropagation with 2nd order derivatives.

The nodes/ contain common implementations. So far we have:
 - fully connected
 - sigmoid / tanh / square activation functions
 
 The tests/ showcase how to set up a network and run forward / backward passes. \
 Currently, the backward pass is done in two separate sweeps. This was easier for testing,
 but will probably be removed for better performance.
 
 The notebooks/ are helper files to be able to compare with analytical derivatives.
 Currently, the tf-grads notebook contains my math notes. But that might be moved here.
 
 The examples/ are broken and need to be updated for my new naming schemes. (wip).