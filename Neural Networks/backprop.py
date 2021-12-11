
import numpy

def backprop(example, w):
    print("hi")


# Inputs
#   w1, w2      : weights to be updated
#   hidden      : hidden (forward value) used for backprop
#   x           : input used for backprop
#   loss_grad   : loss function gradient
#   y_pred      : predicted output
#   grad_y_pred : output gradients
#   lr          : learning rate
# Outputs
#   w1, w2      : updated weights
def backward(w1, w2, w3, hidden1, hidden2, x, y_true,y_pred,lr=1e-4):
        loss_grad = y_pred - y_true
        grad_w3 = loss_grad * hidden2

        grad_upstream = loss_grad * (y_pred * (1 - y_pred))
        grad_w2 = hidden.T.dot(grad_upstream) 
        grad_hidden = grad_upstream.dot(w2.T) 
        grad_w1 = x.T.dot(grad_hidden) 

        # Update and return new weights for the 3 layers
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2
        w3 -= lr * grad_w3
        return w1,w2, w3