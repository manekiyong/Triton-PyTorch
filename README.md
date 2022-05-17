# Triton Setup
 

## 1. Converting Model to Torchscript (By Tracing)
For PyTorch model to be used with Triton, it first needs to be converted into Torchscript model. Refer to `Convert.ipynb` (Courtesy of [Deep Java Library](https://djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html)).

```
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True).cuda()

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("model.pt")
```

Notes:
- Input of model should be of type `torch.Tensor`. Other input types may cause incorrect tracing, or might not work at all. 
- If custom model consist of conditional logics that affects the final output, then the portion must be isolated and scripted instead. Refer to [PyTorch's Mixing Tracing and Scripting](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting) for more details. 
    - Scripting would not be necessary if the model always go through a single logic flow only when deployed.



## 2. Configure Triton Repository
With reference to [Triton's Inference Server Repository](https://github.com/triton-inference-server/server), files need to be organised in the following layout:

```
.              
├── models
│   ├── <model1>
|   |   ├── <version1>
|   |   |   └── <TorchScript .pt file>
|   |   ├── <version2>
|   |   |   └── <TorchScript .pt file>
|   |   └── config.pbtxt
│   └── <model2>
|       ├── <version1>
|       |   └── <TorchScript .pt file>
|       ├── <version2>
|       |   └── <TorchScript .pt file>
|       └── config.pbtxt
└── Dockerfile
```

An example of such repository is shown in the `Triton` folder.

### Key Components of Triton Repository

#### Dockerfile

- Used to build the repository into a docker image. Key thing to adjust here is the version of `tritonserver` base docker image, as necessary.

#### model.pt 
- Copy the `model.pt` generated in the previous step into the appropriate path as shown in the layout above. 


#### config.pbtxt

- Each model requires a `config.pbtxt` file that, most importantly, describes the input and output shape of the model. The basic `config.pbtxt` file for PyTorch is in `Triton\models\facenet` and `Triton\models\mtcnn` folder. 
- `data_type`: For each input and output, the `data_type` needs to specified. The valid `data_type` and their PyTorch equivalent can be found in [Triton's Server Model Configuration Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes)
- `dims`: Specify the dimensions of a single input example. In cases where the dimension varies (e.g. in the case of width and height of images), `-1` will be listed for those dimension (Example shown in `Triton\models\mtcnn\config.pbtxt`)

This repository describes the very basic Triton repository for PyTorch. All other specifications and configurations of `config.pbtxt` (e.g. `version_policy`, `max_batch_size` etc) can be found in the [Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)


## Serving Triton Server

### By Docker Run

### By Docker-Compose


## Inference