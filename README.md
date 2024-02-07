# 因为MB-Taylorformer太难懂了，所以准备自己复现Taylorformer

# Taylorformer是基于tensorflow的，我用不习惯，因此我准备用pytorch复现。

## Taylorformer-tensorflow文件结构：

Our model architecture is shown below:

    ┬─ __init__.py
    ├─ pre_trained_model_ex.py
    ├─ training_and_evaluating_models.py
    ├─ data_wrangler
    │   ├─ __init__.py
    │   ├─ batcher.py
    │   ├─ dataset_preparer.py
    │   └─ feature_extractor.py
    ├─ comparison_models/tnp
    │   ├─ __init__.py
    │   ├─ tnp.py
    │   └─ tnp_pipeline.py
    ├─ model
    │   ├─ __init__.py
    │   ├─ dot_prod.py
    │   ├─ losses.py
    │   ├─ taylorformer.py
    │   ├─ taylorformer_graph.py
    │   └─ taylorformer_pipeline.py
    └─ weights_/forecasting/ETT/taylorformer/96/ckpt
        ├─ check_run_0
        │   ├─ checkpoint
        │   ├─ ckpt-37.data-00000-of-00001
        │   └─ ckpt-37.index
        ├─ check_run_1
        │   ├─ checkpoint
        │   ├─ ckpt-26.data-00000-of-00001
        │   └─ ckpt-26.index
        └─ ... (chekpoint name)

## 网络结构图例：（我没看懂）

<img width="784" alt="image" src="https://github.com/oremnirv/ATP/assets/54116509/7a8f1e82-4f91-4cb2-89ec-748f8556529a">


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training and Evaluation

### Training

To train the model(s) in the paper, run this command:

```train
python training_and_evaluation.py "<type of dataset>" "<model>" num_iterations num_repeat_runs n_C n_T 0
```
where <type of dataset> is for example ETT or exchange, <model> is for example, TNP or taylorformer, where n_C and n_T are the number of context and target points, respectively.
  
You will have needed to create appropriate folders to store the model weights and evaluation metrics. We have included a folder for the taylorformer on the ETT dataset, with n_T = 96, as an example. Its path is `weights_/forecasting/ETT/taylorformer/96`.

### Evaluation 

Evaluation metrics (mse and log-likelihood) for each of the repeat runs are saved in the corresponding folder e.g. `weights_/forecasting/ETT/taylorformer/96`. The mean and standard deviations are used when reporting the results.
  
### Load pre-trained model 

 Here is an example of how to load a pre-trained model for the ETT dataset with the Taylorformer for the target-96-context-96 setting.
  
  
```
python pre_trained_model_ex.py 0 37
```
  
## Results
  
We show our results on the forecasting datasets. More results can be found in the paper.
  
<img width="717" alt="image" src="https://github.com/oremnirv/ATP/assets/54116509/45c9efad-41cb-4ad1-aa16-d643eb8e23ad">

  



