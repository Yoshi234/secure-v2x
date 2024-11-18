# plaintext implementations

we implement 5-6 models (I think this is a good target number)
which provide reasonable representations of implementations 
across time (the development of graph neural networks for the traffic
flow prediction task)

## best candidate models

models are selected base on the following criteria, in order of importance:

1. feasibility: can the model successfully be implemented in existing 
   privacy-preserving frameworks?
2. performance efficiency: few results exist which measure the run time 
   of most relevant models. This will be a result which we need to generate
   indpendently as a point of relevance for model evaluation
3. accuracy / sota performance: the most obvious metric for selecting a model
   under normal circumstances

<center>

model | rating | reason
---|---|---
stgcn | 10.0 | feasible, reasonable performance and efficiency
astgcn | 7.0 | feasible, may need to implement cheb polynomials (not sure what this will entail exactly) - must be implemented to make a final determination
st-resnet | 10.0 | highly feasible, incorporates primarily foundational modules (resnet) which can EASILY be implemented in crypten, and even in more advanced sota frameworks such as cryptflow2 and cheetah - this significantly strengthens the paper
stsgcn | 5.0 | not a particularly interesting work to me, but it might be feasible for implementation - not worth discounting completely yet
graph wavenet | 9.0 | high performing baseline, but not state of the art. main challenge will be rewriting all of the einsum operators in order to get them to work. Additionally, this network uses the diagonal and svd operators, which will need to be developed in order for the network to work. 
gman | 5.0 | good baseline, but again, not super interesting
st-ssl | 10.0 | *SOTA model - highest performance available for the task. The layernorm operation may not be supported, and torch.eye and torch.diag may nott be supported either - may require additional secure implementations
stpgcn | 8.0 | strong model, requires torch.nn.LayerNorm to work however. Not sure if this will work or not (depends on whether the embeddings module can convert or not)
stfgnn | 9.0 | good model, and doesn't use spectral graph convolutions - this might be more efficient, and is almost certainly implementable (easily)

</center>

The plan will be to run these models in the following order to obtain baseline results: 

1. st-resnet
2. stfgnn (pytorch impl. https://github.com/lwm412/STFGNN-Pytorch/blob/main/STFGNN/model/STFGNN.py)
3. st-ssl
4. stgcn
5. graph wavenet
6. stpgcn

Pretrained pytorch models need to be constructed, and saved to a .pth 
file. The 

A repository with most of the datasets is available 
[here](https://github.com/LibCity/Bigscity-LibCity-Datasets?tab=readme-ov-file#Traffic-State-Datasets-Grid-based-In-Flow-and-Out-Flow)

## Model Performance Benchmarks

<center> 

model | dataset | mae | mape
---|---|---|---
ST-SSL | BJTaxi | 11.48 | 16.229%

</center>

