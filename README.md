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
forward_pass1: inp -> out
backward_pass1: dy_dout -> dy_din
forward_pass2: din_dx -> dout_dx

TODO: the second forward pass seems unnecessary

in analogy to the definitions above, we also have the second derivatives
ddy_ddout, ddy_ddin, ddout_ddin, ddy_ddx, ...

if we have ddy_ddout, we can calculate ddy_ddin like this:\
ddy_ddin = ddy_ddout * (dout_din)**2 + dy_dout * ddout_ddin

this is done in the last pass:
backward_pass2: ddy_dout -> ddy_ddin