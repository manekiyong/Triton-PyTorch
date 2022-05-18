# Triton PyTorch
 

## 1. Converting Model to Torchscript (By Tracing)
For PyTorch model to be used with Triton, it first needs to be converted into Torchscript model. Refer to `trace_model.ipynb` for tracing example of ResNet model and BERT model (Courtesy of [Deep Java Library](https://djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html) for ResNet tracing, and [Hugging Face](https://huggingface.co/docs/transformers/serialization#saving-a-model) for BERT tracing).


Notes:
- Input of model during tracing should be of type `torch.Tensor`. Other input types may cause incorrect tracing, or might not work at all. 
- If custom model consist of conditional logics that affects the final output, then the portion must be isolated and scripted instead. Refer to [PyTorch's Mixing Tracing and Scripting](https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting) for more details. 
    - Scripting would not be necessary if the model always go through a single logic flow only when deployed.


## 2. Configure Triton Repository
With reference to [Triton's Inference Server Repository](https://github.com/triton-inference-server/server), files need to be organised in the following layout:

```
.              
├── models
│   ├── <model1>
|   |   ├── <version1>
|   |   |   └── model.pt
|   |   ├── <version2>
|   |   |   └── model.pt
|   |   └── config.pbtxt
│   └── <model2>
|       ├── <version1>
|       |   └── model.pt
|       ├── <version2>
|       |   └── model.pt
|       └── config.pbtxt
└── Dockerfile
```

An example of such repository is shown in the `Triton` folder. 

Note: `.pt` file MUST be renamed as `model.pt`

### Key Components of Triton Repository

#### Dockerfile
- Used to build the repository into a docker image. Key thing to adjust here is the version of `tritonserver` base docker image, as necessary.

#### model.pt 
- Copy the `model.pt` generated in the previous step into the appropriate path as shown in the layout above. 


#### config.pbtxt

- Each model requires a `config.pbtxt` file that, most importantly, describes the input and output shape of the model. The basic `config.pbtxt` file for PyTorch is in `Triton\models\resnet` and `Triton\models\bert` folder. 
- `data_type`: For each input and output, the `data_type` needs to specified. The valid `data_type` and their PyTorch equivalent can be found in [Triton's Server Model Configuration Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes)
- `dims`: Specify the dimensions of a single input example. In cases where the dimension varies (e.g. in the case of width and height of images), `-1` will be listed for those dimension (Example shown in `Triton\models\bert\config.pbtxt`, where texts have varying sequence length)

The example in this repository describes the very basic Triton repository for PyTorch. All other specifications and configurations of `config.pbtxt` (e.g. `version_policy`, `max_batch_size` etc) can be found in the [Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)


## Serving Triton Server

### By Docker Run
Execute the following command to run tritonserver as a standalone service. In the following example, only 1 GPU is used, and version `22.02-py3` of `tritonserver` is used.
- `docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/Triton/models:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models`

### By Docker-Compose
Alternatively, you may used `docker-compose.yml` to run tritonserver alongside other services (e.g. FastAPI, ElasticSearch etc), and configure the necessary links and volumes.
- `docker-compose up`

When the triton server is successfully served, you should see the following:
```
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| bert   | 1       | READY  |
| resnet | 1       | READY  |
+--------+---------+--------+

```

## Inference

A skeleton inference code can be found in `infer.ipynb`. Code is adapted from [Chronicles of AI](https://chroniclesofai.com/mlops-chapter-8-model-server-with-nvidia-triton-local-part-1-b/). 

Within the inference code, again the data type needs to be specified within the `httpclient.InferInput()` function for each input. Refer again to [Triton's Server Model Configuration Documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes), but use the API column as the datatype instead. 

