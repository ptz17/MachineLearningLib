#import tensorflow as tf
import onnx
import onnxmltools
import torch
def Torch2Light(Model, input_dim, nClasses, batch_size, path):
    # Load your Torch model and convert to onnx model
    x = torch.randn(input_dim, requires_grad=True)
    torch.onnx.export(Model, x, path,
		input_names = ['input'], output_names = ['output'], 
		dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print("Computation (onnx) graphs has %d nodes" % len(onnx_model.graph.node))
    for i in range(len(onnx_model.graph.node)):
        print("node type: %s" % onnx_model.graph.node[i].op_type)
    return onnx
