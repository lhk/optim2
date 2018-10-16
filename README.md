# optim2
playing with 2nd order optimization schemes


each individual layer calculates inp -> out
the local derivatives are ddout, the derivative of the output wrt the input
the model(chain of all layers) calculates x -> y
the chain looks like this:\
x -> ... -> inp --(our layer)--> out -> ... -> y
which leads to the definition of the following derivatives
dy_dout (derivative of model output wrt our layer's output)
dy_din (derivative of model output wrt our layers's input, this includes the calculation of our layer)
din_dx (derivative of layer input wrt model input)
dout_dx (derivative of layer output wrt model input)

since the layers are chained, these names are used in two ways:
dout_dx in our layer is the same as din_dx in the next layer

we eventually need all these to calculate the second derivatives
to do this, I use the following functions:
forward: inp -> out
backward: dy_dout -> dy_din
forward_grad: din_dx -> dout_dx

in analogy to the definitions above, we also have
ddy_ddx, ddy_ddout, ...
the second derivative at our specific layer contributes like this:
dy_dout * ddout_ddin * (din_dx)**2

this is hard to write down properly
we receive an inflowing second gradient
this we multiply with our own first gradient and add our own second
