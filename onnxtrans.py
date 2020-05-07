import onnx
import math

input_size =(1088,1920)
model = onnx.load_model("./models/onnx/centerface.onnx")
d = model.graph.input[0].type.tensor_type.shape.dim
print(d)
rate = (int(math.ceil(input_size[0]/d[2].dim_value)),int(math.ceil(input_size[1]/d[3].dim_value)))
print("rare",rate)
d[0].dim_value = 1
d[2].dim_value *= rate[0]
d[3].dim_value *= rate[1]
for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    print(d)
    d[0].dim_value = 1
    d[2].dim_value  *= rate[0]
    d[3].dim_value  *= rate[1]

onnx.save_model(model,"./models/onnx/enterface_1088_1920.onnx" )