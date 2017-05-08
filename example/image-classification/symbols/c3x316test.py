"""
a simple multilayer perceptron
"""
import mxnet as mx

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    c3x3  = mx.symbol.c3x3_16(data = data,  num_filter=4, kernel=(3,3), name='c3x3')
    flatten = mx.symbol.Flatten(data=c3x3)
    fc1  = mx.symbol.FullyConnected(data = flatten, name='fc1', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')
    return mlp
