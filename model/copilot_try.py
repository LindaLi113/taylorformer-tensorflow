Here is the tensorflow project:
1. Organization of the project:
┬─ README.md
├─ __init__.py
├─ pre_trained_model_ex.py
├─ training_and_evaluating_models.py
├─ data_wrangler
│  ├─ __init__.py
│  ├─ batcher.py
│  ├─ dataset_preparer.py
│  └─ feature_extractor.py
├─ comparison_models/tnp
│  ├─ __init__.py
│  ├─ tnp.py
│  └─ tnp_pipeline.py
├─ model
│  ├─ __init__.py
│  ├─ dot_prod.py
│  ├─ losses.py
│  ├─ taylorformer.py
│  ├─ taylorformer_graph.py
│  └─ taylorformer_pipeline.py
└─ weights_/forecasting/ETT/taylorformer/96/ckpt
  ├─ check_run_0
  │  ├─ checkpoint
  │  ├─ ckpt-37.data-00000-of-00001
  │  └─ ckpt-37.index
  └─ ... (chekpoint name)

2. 项目代码：
(1) README.md:
 
(2) __init__.py: nothing

(3) pre_trained_model_ex.py:
from data_wrangler import dataset_preparer
from model import taylorformer_pipeline
import numpy as np
import tensorflow as tf
from data_wrangler.batcher import batcher
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("run",type=int,help = "run number", default=0) # change this to the run number you want to use [0, 1, 2, 3, 4] 
  parser.add_argument("step",type=int,help = "step number", default=37) # change this to the step number given in the checkpoint file end e.g. ckpt-37 is inside weights_/forecasting/ETT/taylorformer/96/ckpt/check_run_0

  args = parser.parse_args()

  n_C = 96
  n_T = 96
  model = 'taylorformer'
  x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
  save_dir = "weights/forecasting/ETT"
  save_dir = save_dir + "/" + model + '/' + str(n_T)
   
  model = taylorformer_pipeline.instantiate_taylorformer('ETT')

  name_comp = 'run_' + str(args.run)
  folder = save_dir + '/ckpt/check_' + name_comp
  opt = tf.keras.optimizers.Adam(3e-4)
     
  ### LOAD THE MODEL
  ckpt = tf.train.Checkpoint(step=tf.Variable(args.step), optimizer=opt, net=model)
  manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
  ckpt.restore(manager.latest_checkpoint) 

  ## Run pre-trained model on batch of test data 
  test_batch_s = 16
  idx_list = list(range(x_test.shape[0] - (n_C + n_T)))
  t_te, y_te, idx_list = batcher(x_test, y_test, idx_list, batch_s = test_batch_s, window=n_C+n_T)
  t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=y_te.shape[0])
  μ, log_σ = model([t_te, y_te, n_C, n_T, False])

(4) training_and_evaluating_models.py:
#!/usr/bin/env python

from model import taylorformer_graph, losses
from data_wrangler import synthetic_data_gen, feature_extractor
import keras
import numpy as np
import tensorflow as tf
from model import taylorformer_pipeline
from comparison_models.tnp import tnp_pipeline
from data_wrangler import dataset_preparer
import argparse
from data_wrangler.batcher import batcher, batcher_np
import os

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help="dataset")
  parser.add_argument("model", type=str, help="model")
  parser.add_argument("iterations", type=int, help="number of iterations for training")
  parser.add_argument("num_repeats", type=int, help="number of random seed repeats")
  parser.add_argument("n_C",type=int,help = "context")
  parser.add_argument("n_T",type=int,help = "target")
  parser.add_argument("run",type=int,help = "run number")
  args = parser.parse_args()

  if args.dataset == "exchange":
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/exchange.csv") 
    save_dir = "weights/forecasting/exchange"
    print('make sure to create the exchange folder in weights/forecasting/')
   
  elif args.dataset == "ETT":
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.dataset_processor(path_to_data="datasets/ETTm2.csv") 
    save_dir = "weights/forecasting/ETT"
    print('make sure to create the ETT folder in weights/forecasting/')

  elif args.dataset == "rbf":
    x_train, y_train, x_val, y_val, x_test, y_test,n_context_test = dataset_preparer.gp_data_processor(path_to_data_folder="datasets/rbf/")
    save_dir = "weights/gp/rbf"
    print('make sure to create the gp folder in weights/gp')

  elif args.dataset == "electricity":
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_preparer.electricity_processor(path_to_data="datasets/electricity_training.npy")
    save_dir = "weights/forecasting/electricity"
    print('make sure to create the electricity folder in weights/forecasting/')

  else: 
    raise ValueError("Dataset not found")
   
  if args.dataset == "rbf":
    save_dir = save_dir + '/' + args.model

  else:
    #### for forecasting inlude the n_T in the save directory
    save_dir = save_dir + "/" + args.model + '/' + str(args.n_T)

  n_C = args.n_C
  n_T = args.n_T

  batch_size = 32
  test_batch_s = 100
  valid_batch_size = 100
  if args.dataset != "rbf":
    if n_T > 700 :
      batch_size = 16
      test_batch_s = 16
      valid_batch_size = 16   

  nll_list = []
  mse_list = []

  for repeat in range(args.num_repeats):

    step = 1
    run= args.run + repeat
    tf.random.set_seed(run)

    if args.model == "taylorformer":
      model = taylorformer_pipeline.instantiate_taylorformer(args.dataset)
     
    if args.model == "tnp":
      model = tnp_pipeline.instantiate_tnp(args.dataset)
   
    tr_step = taylorformer_graph.build_graph()
     
    ###### can we put the name of the model into the folder name #########?

    name_comp = 'run_' + str(run)
    folder = save_dir + '/ckpt/check_' + name_comp
    if not os.path.exists(folder): os.mkdir(folder)
    opt = tf.keras.optimizers.Adam(3e-4)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint) 
    sum_mse_tot = 0; sum_nll_tot = 0
    mini = 50000

    validation_losses = []

    for i in range(args.iterations):

      if args.dataset != "rbf":
        idx_list = list(range(x_train.shape[0] - (n_C+n_T)))
        x,y,_ = batcher(x_train,y_train,idx_list,window=n_C+n_T,batch_s=batch_size) ####### generalise for not just forecasting
        x = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:, np.newaxis], axis=0, repeats=batch_size) # it doesnt matter what the time is, just the relation between the times.
        #### edit batcher to fix this

      #### batcher for NP data (which is already in sequences)
      if args.dataset == "rbf":
        x,y = batcher_np(x_train,y_train,batch_s=batch_size)
        #### nc nT need specification
        n_C = tf.constant(int(np.random.choice(np.linspace(3,99,97))))
        n_T = 100 - n_C

      _,_, _, _ = tr_step(model, opt, x,y,n_C,n_T, training=True)

      if i % 100 == 0:

        if args.dataset != "rbf":
          idx_list = list(range(x_val.shape[0] - (n_C+n_T)))
          ##### if val set is empty - increase val set size. ########
          t_te,y_te,_ = batcher(x_val,y_val,idx_list,batch_s = valid_batch_size,window=n_C+n_T)
          t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=valid_batch_size)

        if args.dataset == "rbf":
          t_te,y_te = batcher_np(x_val,y_val,batch_s=valid_batch_size)
          ####nc nt need to be specified without interfering with the nc nt above
          n_C = 20
          n_T = 80
        μ, log_σ = model([t_te, y_te, n_C, n_T, False])
        _,_,_, nll_pp_te, msex_te = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)

        validation_losses.append(nll_pp_te)

        np.save(folder + "/validation_losses_iteration",np.array(validation_losses))

        if nll_pp_te < mini:
          mini = nll_pp_te
          manager.save()
          step += 1
          ckpt.step.assign_add(1)

    ######### Evaluation code ################ 
    ckpt = tf.train.Checkpoint(step=tf.Variable(step), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint) 

    if args.dataset != "rbf":
  
      test_batch_s = 100 #need to specify this as it gets changed in the loop below
      if n_T > 700 :
        test_batch_s = 16
      idx_list = list(range(x_test.shape[0] - (n_C+n_T)))
      num_batches = len(idx_list)//test_batch_s

      for _ in range(num_batches): #### specify correct number of batches for the batcher #####
        if(_ == (num_batches-1)): test_batch_s = len(idx_list)     
        t_te,y_te,idx_list = batcher(x_test, y_test, idx_list,batch_s = test_batch_s, window=n_C+n_T)
        t_te = np.repeat(np.linspace(-1,1,(n_C+n_T))[np.newaxis,:,np.newaxis],axis=0,repeats=y_te.shape[0])
        μ, log_σ = model([t_te, y_te, n_C, n_T, False])
        _, sum_mse, sum_nll, _, _ = losses.nll(y_te[:, n_C:n_C+n_T], μ, log_σ)
        sum_nll_tot += sum_nll / n_T
        sum_mse_tot += sum_mse / n_T

      nllx = sum_nll_tot / (test_batch_s * x_test.shape[0]//test_batch_s)
      msex = sum_mse_tot / (test_batch_s * x_test.shape[0]//test_batch_s)

    if args.dataset == "rbf":

      for i in range(x_test.shape[0]):
        n_C = tf.constant(n_context_test[i],"int32")
        n_T = 100 - n_C
        μ, log_σ = model([x_test[i:i+1], y_test[i:i+1], n_C, n_T, False])
        _, sum_mse, sum_nll, _, _ = losses.nll(y_test[i:i+1, n_C:n_C+n_T], μ, log_σ)
        sum_nll_tot += sum_nll / n_T
        sum_mse_tot += sum_mse / n_T

      nllx = sum_nll_tot / x_test.shape[0]
      msex = sum_mse_tot / x_test.shape[0]

    nll_list.append(nllx.numpy())
    mse_list.append(msex.numpy())        
       
    np.save(save_dir + '/nll_list.npy', nll_list)   
    np.save(save_dir + '/mse_list.npy', mse_list)  

(5) comparison_models/tnp/__init__.py: nothing

(6) comparison_models/tnp/tnp.py:
import tensorflow as tf
import sys
sys.path.append("../../")
from model import dot_prod

class FFN(tf.keras.layers.Layer):
  def __init__(self, output_shape, dropout_rate=0.1):
    super().__init__()

    self.dense_b = tf.keras.layers.Dense(output_shape)
    self.dense_c = tf.keras.layers.Dense(output_shape)
    self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]     
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, query):

   ## query is the output of previous MHA_X layer
   ## x is query input to MHA_X_o 

    x += query
    x = self.layernorm[0](x)
    x_skip = tf.identity(x)
    x = self.dense_b(x)
    x = tf.nn.gelu(x)
    x = self.dropout(x)
    x = self.dense_c(x)
    x += x_skip
    return self.layernorm[1](x)

class MHA_XY(tf.keras.layers.Layer):
  def __init__(self,
         num_heads,
         projection_shape,
         output_shape,
         dropout_rate=0.1):
    super().__init__()
    self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
    self.ffn = FFN(output_shape, dropout_rate)

  def call(self, query, key, value, mask):
    x = self.mha(query, key, value, mask)
    x = self.ffn(x, query) # Shape `(batch_size, seq_len, output_shape)`.
    return x
  
class embed_layers(tf.keras.layers.Layer):
  def __init__(self,output_shape,num_layers_embed=4):
    super().__init__()
    self.num_layers = num_layers_embed
    self.embed = [tf.keras.layers.Dense(output_shape,activation="relu") for _ in range(self.num_layers-1)]
    self.embed.append(tf.keras.layers.Dense(output_shape))

  def call(self,inputs):
    x = inputs
    for i in range(self.num_layers):
      x = self.embed[i](x)
    return x

class TNP_Decoder(tf.keras.models.Model):
  def __init__(self,output_shape=64,num_layers=6,projection_shape=16,
         num_heads=4,dropout_rate=0.0,target_y_dim=1,bound_std=False):
    super().__init__()

    self.num_layers = num_layers

    self.mha_xy = [MHA_XY(num_heads,projection_shape,
               output_shape,dropout_rate) for _ in range(num_layers)]

    self.embed = embed_layers(output_shape,num_layers_embed=4)

    self.dense = tf.keras.layers.Dense(output_shape,activation="relu")
    self.linear = tf.keras.layers.Dense(2*target_y_dim)
    self.target_y_dim = target_y_dim
    self.bound_std = bound_std
     
  def call(self,inputs,training=True):
     
    context_target_pairs,target_masked_pairs,mask = inputs
    input_for_mha = tf.concat([context_target_pairs,target_masked_pairs],axis=1)

    embed = self.embed(input_for_mha)
     
    v = embed
    k = tf.identity(v)
    q = tf.identity(v)

    for i in range(self.num_layers):
      x = self.mha_xy[i](q,k,v,mask)
      q = tf.identity(x)
      k = tf.identity(x)
      v = tf.identity(x)
    
    L = self.dense(x)
    L = self.linear(L)

    mean,log_sigma = L[:,:,:self.target_y_dim],L[:,:,self.target_y_dim:]

    if self.bound_std:
      sigma = 0.05 + 0.95 * tf.math.softplus(log_sigma)
    else:
      sigma = tf.exp(log_sigma)
     
    log_sigma = tf.math.log(sigma)
    return mean,log_sigma      

(7) comparison_models/tnp/tnp_pipeline.py:
import tensorflow as tf
from tensorflow import keras
from comparison_models.tnp.tnp import TNP_Decoder
import sys
sys.path.append("../../")
from data_wrangler.feature_extractor import feature_wrapper

class tnp_pipeline(keras.models.Model):

  def __init__(self,num_heads=4,projection_shape_for_head=4,output_shape=64, dropout_rate=0.0, 
         permutation_repeats=1,bound_std=False, num_layers=6,target_y_dim=1):
    super().__init__()
    self._permutation_repeats = permutation_repeats
    self._feature_wrapper = feature_wrapper()
    self._tnp = TNP_Decoder(output_shape=output_shape,num_layers=num_layers,projection_shape=int(projection_shape_for_head*num_heads),
        num_heads=num_heads,dropout_rate=dropout_rate,target_y_dim=target_y_dim,bound_std=bound_std)

  def call(self, inputs):

    x, y, n_C, n_T, training = inputs
    #x and y have shape batch size x length x dim

    x = x[:,:n_C+n_T,:]
    y = y[:,:n_C+n_T,:]

    if training == True:   
      x,y = self._feature_wrapper.permute([x, y, n_C, n_T, self._permutation_repeats]) 

    ######## make mask #######

    context_part = tf.concat([tf.ones((n_C,n_C),tf.bool),tf.zeros((n_C,2*n_T),tf.bool)],
             axis=-1)
    first_part = tf.linalg.band_part(tf.ones((n_T,n_C+2*n_T),tf.bool),-1,n_C)
    second_part = tf.linalg.band_part(tf.ones((n_T,n_C+2*n_T),tf.bool),-1,n_C-1)
    mask = tf.concat([context_part,first_part,second_part],axis=0)
     
    ###### mask appropriate inputs ######

    batch_s = tf.shape(x)[0]

    context_target_pairs = tf.concat([x,y],axis=2)
     
    y_masked = tf.zeros((batch_s,n_T,y.shape[-1]))
    target_masked_pairs = tf.concat([x[:,n_C:],y_masked],axis=2)

    μ, log_σ = self._tnp([context_target_pairs,target_masked_pairs,mask],training)
    return μ[:,-n_T:], log_σ[:, -n_T:]      

def instantiate_tnp(dataset,training=True):
     
  if dataset == "exchange":

    return tnp_pipeline(num_heads=6,projection_shape_for_head=8,output_shape=48, dropout_rate=0.1, 
         permutation_repeats=0,bound_std=False, num_layers=6,target_y_dim=1)

  if dataset == "ETT":

    return tnp_pipeline(num_heads=7,projection_shape_for_head=12,output_shape=48, dropout_rate=0.05, 
         permutation_repeats=0,bound_std=False, num_layers=4,target_y_dim=1)

(8) data_wrangler/__init__.py: nothing

(9) data_wrangler/batcher.py：
import numpy as np

def batcher(t, y, idx_list, batch_s = 32, window = 288):
  '''
  cutting one long array to sequences of length 'window'.
  'batch_s' must be ≤ full array - window length

  input to forecast: (None, 1, 1) for t,y.
  input to NP tasks: (None, seq_len, 1) for t,y. window = 1.
  idx_list: list of indices, must be ≤ full array - window length.
  '''
   
  if len(idx_list) < 1:
    print("warning- you didn't loop over the correct range")
     
  batch_s = min(batch_s, y.shape[0]-window)   
  idx = np.random.choice(len(idx_list), batch_s, replace = False)

  y = np.array([np.array(y)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
  t = np.array([np.array(t)[idx_list[i]:idx_list[i]+window, :, :] for i in idx])
  for i in sorted(idx, reverse=True): del idx_list[i]
     
  t = t.squeeze()
  y = y.squeeze()
   
  if len(t.shape) == 2:
    t = t[:,:,np.newaxis]
    y = y[:,:,np.newaxis]
     
  return t,y, idx_list

def batcher_np(t,y,batch_s=32):

  idx = np.random.choice(y.shape[0], batch_s, replace = False)

  y = y[idx, :, :]
  t = t[idx, :, :]

  return t,y

(10) data_wrangler/dataset_preparer.py:
import numpy as np
import tensorflow as tf
import pandas as pd

def dataset_processor(path_to_data):
    # works for exchange and ETTm2 dataset w/o extra features 

    pd_array = pd.read_csv(path_to_data)
    data = np.array(pd_array)
    data[:,0] = np.linspace(-1,1,data.shape[0])
    # we need to have it between -1 to 1 for each batch item not just overall!!!!!!!!

    data = data.astype("float32")

    training_data = data[:int(0.69*data.shape[0])]
    val_data = data[int(0.69*data.shape[0]):int(0.8*data.shape[0])]
    test_data = data[int(0.8*data.shape[0]):]

    #scale

    training_data_scaled = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
    val_data_scaled = (val_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
    test_data_scaled = (test_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)

    x_train, y_train = training_data_scaled[:,:1], training_data_scaled[:,-1:]
    x_val, y_val = val_data_scaled[:,:1], val_data_scaled[:,-1:]
    x_test, y_test = test_data_scaled[:,:1], test_data_scaled[:,-1:]

    return x_train[:,:,np.newaxis], y_train[:,:,np.newaxis], x_val[:,:,np.newaxis], y_val[:,:,np.newaxis], x_test[:,:,np.newaxis], y_test[:,:,np.newaxis]

def electricity_processor(path_to_data):
    data = np.load(path_to_data)
    data = (data[322:323].transpose([1,0])) #shape 14000 x 1
     
    time = np.linspace(-1,1,data.shape[0]) # shape 14000
    time = time[:,np.newaxis] # shape 14000 x 1

    data = np.concatenate([time,data],axis=1) # shape 14000 x 2

    training_data = data[:int(0.69*data.shape[0])]
    val_data = data[int(0.69*data.shape[0]):int(0.8*data.shape[0])]
    test_data = data[int(0.8*data.shape[0]):]

    #scale
    training_data_scaled = (training_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
    val_data_scaled = (val_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)
    test_data_scaled = (test_data - np.mean(training_data,axis=0))/np.std(training_data,axis=0)

    x_train, y_train = training_data_scaled[:,:1], training_data_scaled[:,-1:]
    x_val, y_val = val_data_scaled[:,:1], val_data_scaled[:,-1:]
    x_test, y_test = test_data_scaled[:,:1], test_data_scaled[:,-1:]

    return x_train[:,:,np.newaxis], y_train[:,:,np.newaxis], x_val[:,:,np.newaxis], y_val[:,:,np.newaxis], x_test[:,:,np.newaxis], y_test[:,:,np.newaxis]

def gp_data_processor(path_to_data_folder):

    x = np.load(path_to_data_folder + "x.npy")
    y = np.load(path_to_data_folder + "y.npy")

    x_train = x[:int(0.99*x.shape[0])]
    y_train = y[:int(0.99*y.shape[0])]
    x_val = x[int(0.99*x.shape[0]):]
    y_val = y[int(0.99*y.shape[0]):]

    x_test = np.load(path_to_data_folder + "x_test.npy")
    y_test = np.load(path_to_data_folder + "y_test.npy")

    context_n_test = np.load(path_to_data_folder + "context_n_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test, context_n_test

(11) data_wrangler/feature_extractor.py:
import numpy as np
import tensorflow as tf
   
class feature_wrapper(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, inputs):
   
    x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T = inputs ##think about clearer notation
     
    dim_x = x_n.shape[-1]
    ##### inputs for the MHA-X head ######
    value_x = tf.identity(y) #check if identity is needed
    x_prime = tf.concat([x_emb, x_diff, x_n], axis=2) ### check what is happening with embedding
    query_x = tf.identity(x_prime)
    key_x = tf.identity(x_prime)

    ##### inputs for the MHA-XY head ######
    y_prime = tf.concat([y, y_diff, d, y_n], axis=-1)
    batch_s = tf.shape(y_prime)[0]
    key_xy_label = tf.zeros((batch_s, n_C+n_T, 1))
    value_xy = tf.concat([y_prime, key_xy_label, x_prime], axis=-1)
    key_xy = tf.identity(value_xy)

    query_xy_label = tf.concat([tf.zeros((batch_s, n_C, 1)), tf.ones((batch_s, n_T, 1))], axis=1)
    y_prime_masked = tf.concat([self.mask_target_pt([y, n_C, n_T]), self.mask_target_pt([y_diff, n_C, n_T]), self.mask_target_pt([d, n_C, n_T]), y_n], axis=2)

    query_xy = tf.concat([y_prime_masked, query_xy_label, x_prime], axis=-1)

    return query_x, key_x, value_x, query_xy, key_xy, value_xy

  def mask_target_pt(self, inputs):
    y, n_C, n_T = inputs
    dim = y.shape[-1]
    batch_s = y.shape[0]

    mask_y = tf.concat([y[:, :n_C], tf.zeros((batch_s, n_T, dim))], axis=1)
    return mask_y
   
  def permute(self, inputs):

    x, y, n_C, _, num_permutation_repeats = inputs

    if (num_permutation_repeats < 1):
      return x, y
     
    else: 
      # Shuffle traget only. tf.random.shuffle only works on the first dimension so we need tf.transpose.
      x_permuted = tf.concat([tf.concat([x[:, :n_C, :], tf.transpose(tf.random.shuffle(tf.transpose(x[:, n_C:, :], perm=[1, 0, 2])), perm =[1, 0, 2])], axis=1) for j in range(num_permutation_repeats)], axis=0)       
      y_permuted = tf.concat([tf.concat([y[:, :n_C, :], tf.transpose(tf.random.shuffle(tf.transpose(y[:, n_C:, :], perm=[1, 0, 2])), perm =[1, 0, 2])], axis=1) for j in range(num_permutation_repeats)], axis=0)

      return x_permuted, y_permuted
         
  def PE(self, inputs): # return.shape=(T, B, d)
    """
    # t.shape=(T, B)  T=sequence_length, B=batch_size
    A position-embedder, similar to the Attention paper, but tweaked to account for
    floating point positions, rather than integer.
    """
    x, enc_dim, xΔmin, xmax = inputs

    R = xmax / xΔmin * 100
    drange_even = tf.cast(xΔmin * R**(tf.range(0, enc_dim, 2) / enc_dim), "float32")
    drange_odd = tf.cast(xΔmin * R**((tf.range(1, enc_dim, 2) - 1) / enc_dim), "float32")
    x = tf.concat([tf.math.sin(x / drange_even), tf.math.cos(x / drange_odd)], axis=2)
    return x       

class DE(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.batch_norm_layer = tf.keras.layers.BatchNormalization()

  def call(self, inputs):
    y, x, n_C, n_T, training = inputs

    if (x.shape[-1] == 1):
      y_diff, x_diff, d, x_n, y_n = self.derivative_function([y, x, n_C, n_T])
    else: 
      y_diff, x_diff, d, x_n, y_n = self.derivative_function_2d([y, x, n_C, n_T])

    d_1 = tf.where(tf.math.is_nan(d), 10000.0, d)
    d_2 = tf.where(tf.abs(d) > 200., 0., d)
    d = self.batch_norm_layer(d_2, training=training)

    d_label = tf.cast(tf.math.equal(d_2, d_1), "float32")
    d = tf.concat([d, d_label], axis=-1)

    return y_diff, x_diff, d, x_n, y_n

###### i think here what we do is calculate the derivative at the given y value and add that in as a feature. This is masked when making predictions
# so the derivative of other y values are what are seen
# Based on taylor expansion, a better feature would be including the derivative of the closest x point, where only seen y values are used for the differencing. 
#this derivative wouldn't need masking.

 ############ check what to do for 2d derivatives - should y diff just be for one point? for residual trick that would make most sense.
 #### but you need mutli-dimensional y for the derivative

 ###### and explain why we do this

  def derivative_function(self, inputs):
     
    y_values, x_values, context_n, target_m = inputs

    epsilon = 0.000002 

    batch_size = y_values.shape[0]

    dim_x = x_values.shape[-1]
    dim_y = y_values.shape[-1]

    #context section

    current_x = tf.expand_dims(x_values[:, :context_n], axis=2)
    current_y = tf.expand_dims(y_values[:, :context_n], axis=2)

    x_temp = x_values[:, :context_n]
    x_temp = tf.repeat(tf.expand_dims(x_temp, axis=1), axis=1, repeats=context_n)
     
    y_temp = y_values[:, :context_n]
    y_temp = tf.repeat(tf.expand_dims(y_temp, axis=1), axis=1, repeats=context_n)
     
    ix = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 1]     
    selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n), 1), (-1, 1)), 
                    tf.reshape(ix, (-1, 1))], axis=1)

    x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, context_n, dim_x)), selection_indices), 
                (batch_size, context_n, dim_x)) 
         
    y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, context_n, dim_y)), selection_indices), 
            (batch_size, context_n, dim_y))
     
    x_rep = current_x[:, :, 0] - x_closest
    y_rep = current_y[:, :, 0] - y_closest       

    deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep, axis=-1, keepdims=True))

    dydx_dummy = deriv
    diff_y_dummy = y_rep
    diff_x_dummy =x_rep
    closest_y_dummy = y_closest
    closest_x_dummy = x_closest

    #target selection

    current_x = tf.expand_dims(x_values[:, context_n:context_n+target_m], axis=2)
    current_y = tf.expand_dims(y_values[:, context_n:context_n+target_m], axis=2)

    x_temp = tf.repeat(tf.expand_dims(x_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)
    y_temp = tf.repeat(tf.expand_dims(y_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)

    x_mask = tf.linalg.band_part(tf.ones((target_m, context_n + target_m), tf.bool), -1, context_n)
    x_mask_inv = (x_mask == False)
    x_mask_float = tf.cast(x_mask_inv, "float32")*1000
    x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float, axis=0), axis=0, repeats=batch_size)
    ix = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                      axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 1]

    selection_indices = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m), 1), (-1, 1)), 
                  tf.reshape(ix, (-1, 1))], axis=1)

    x_closest = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices), 
                (batch_size, target_m, dim_x)) 
     
    y_closest = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices), 
            (batch_size, target_m, dim_y))
         
    x_rep = current_x[:, :, 0] - x_closest
    y_rep = current_y[:, :, 0] - y_closest       

    deriv = y_rep / (epsilon + tf.math.reduce_euclidean_norm(x_rep, axis=-1, keepdims=True))

    dydx_dummy = tf.concat([dydx_dummy, deriv], axis=1)
    diff_y_dummy = tf.concat([diff_y_dummy, y_rep], axis=1)
    diff_x_dummy = tf.concat([diff_x_dummy, x_rep], axis=1)
    closest_y_dummy = tf.concat([closest_y_dummy, y_closest], axis=1)
    closest_x_dummy = tf.concat([closest_x_dummy, x_closest], axis=1)

    return diff_y_dummy, diff_x_dummy, dydx_dummy, closest_x_dummy, closest_y_dummy

  def derivative_function_2d(self, inputs):

      epsilon = 0.0000
     
      def dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2):
        #"z" is the second dim of x input
        numerator = y_closest_2 - current_y[:, :, 0] - ((x_closest_2[:, :, :1]-current_x[:, :, 0, :1])*(y_closest_1-current_y[:, :, 0] ))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1] +epsilon)
        denom = x_closest_2[:, :, 1:2] - current_x[:, :, 0, 1:2] - (x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2])*(x_closest_2[:, :, :1]-current_x[:, :, 0, :1])/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
        dydz_pred = numerator/(denom+epsilon)
        return dydz_pred
       
      def dydx(dydz, current_y, y_closest_1, current_x, x_closest_1):
        dydx = (y_closest_1-current_y[:, :, 0] - dydz*(x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2]))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
        return dydx

      y_values, x_values, context_n, target_m = inputs

      batch_size, length = y_values.shape[0], context_n + target_m

      dim_x = x_values.shape[-1]
      dim_y = y_values.shape[-1]

      #context section

      current_x = tf.expand_dims(x_values[:, :context_n], axis=2)
      current_y = tf.expand_dims(y_values[:, :context_n], axis=2)

      x_temp = x_values[:, :context_n]
      x_temp = tf.repeat(tf.expand_dims(x_temp, axis=1), axis=1, repeats=context_n)

      y_temp = y_values[:, :context_n]
      y_temp = tf.repeat(tf.expand_dims(y_temp, axis=1), axis=1, repeats=context_n)

      ix_1 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 1]     
      selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n), 1), (-1, 1)), 
                        tf.reshape(ix_1, (-1, 1))], axis=1)

      ix_2 = tf.argsort(tf.math.reduce_euclidean_norm((current_x - x_temp), axis=-1), axis=-1)[:, :, 2]     
      selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*context_n), 1), (-1, 1)), 
                    tf.reshape(ix_2, (-1, 1))], axis=1)

      x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_1), 
                (batch_size, context_n, dim_x)) +  tf.random.normal(shape=(batch_size, context_n, dim_x), stddev=0.01)

      x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_2), 
                (batch_size, context_n, dim_x)) +  tf.random.normal(shape=(batch_size, context_n, dim_x), stddev=0.01)

      y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_1), 
            (batch_size, context_n, dim_y))

      y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_2), 
            (batch_size, context_n, dim_y))

      x_rep_1 = current_x[:, :, 0] - x_closest_1
      x_rep_2 = current_x[:, :, 0] - x_closest_2

      print(y_closest_1.shape, current_y.shape)
      y_rep_1 = current_y[:, :, 0] - y_closest_1
      y_rep_2 = current_y[:, :, 0] - y_closest_2

      dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
      dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)

      deriv_dummy = tf.concat([dydx_1, dydx_2], axis=-1)

      diff_y_dummy = tf.concat([y_rep_1, y_rep_2], axis=-1)

      diff_x_dummy =tf.concat([x_rep_1, x_rep_2], axis=-1)

      closest_y_dummy = tf.concat([y_closest_1, y_closest_2], axis=-1)
      closest_x_dummy = tf.concat([x_closest_1, x_closest_2], axis=-1)

      #target selection

      current_x = tf.expand_dims(x_values[:, context_n:context_n+target_m], axis=2)
      current_y = tf.expand_dims(y_values[:, context_n:context_n+target_m], axis=2)

      x_temp = tf.repeat(tf.expand_dims(x_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)
      y_temp = tf.repeat(tf.expand_dims(y_values[:, :target_m+context_n], axis=1), axis=1, repeats=target_m)

      x_mask = tf.linalg.band_part(tf.ones((target_m, context_n + target_m), tf.bool), -1, context_n)
      x_mask_inv = (x_mask == False)
      x_mask_float = tf.cast(x_mask_inv, "float32")*1000
      x_mask_float_repeat = tf.repeat(tf.expand_dims(x_mask_float, axis=0), axis=0, repeats=batch_size)
             
      ix_1 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                        axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 1]
      selection_indices_1 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m), 1), (-1, 1)), 
                        tf.reshape(ix_1, (-1, 1))], axis=1)     
     
      ix_2 = tf.argsort(tf.cast(tf.math.reduce_euclidean_norm((current_x - x_temp), 
                        axis=-1), dtype="float32") + x_mask_float_repeat, axis=-1)[:, :, 2]
      selection_indices_2 = tf.concat([tf.reshape(tf.repeat(tf.range(batch_size*target_m), 1), (-1, 1)), 
                        tf.reshape(ix_2, (-1, 1))], axis=1)
               
      x_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices_1), 
                (batch_size, target_m, dim_x)) +  tf.random.normal(shape=(batch_size, target_m, dim_x), stddev=0.01)

      x_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices_2), 
                (batch_size, target_m, dim_x)) +  tf.random.normal(shape=(batch_size, target_m, dim_x), stddev=0.01)



      y_closest_1 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices_1), 
            (batch_size, target_m, dim_y))

      y_closest_2 = tf.reshape(tf.gather_nd(tf.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices_2), 
            (batch_size, target_m, dim_y))
     
      x_rep_1 = current_x[:, :, 0] - x_closest_1
      x_rep_2 = current_x[:, :, 0] - x_closest_2

      y_rep_1 = current_y[:, :, 0] - y_closest_1
      y_rep_2 = current_y[:, :, 0] - y_closest_2

      dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
      dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)
       
      deriv_dummy_2 = tf.concat([dydx_1, dydx_2], axis=-1)

      diff_y_dummy_2 = tf.concat([y_rep_1, y_rep_2], axis=-1)

      diff_x_dummy_2 =tf.concat([x_rep_1, x_rep_2], axis=-1)

      closest_y_dummy_2 = tf.concat([y_closest_1, y_closest_2], axis=-1)
      closest_x_dummy_2 = tf.concat([x_closest_1, x_closest_2], axis=-1)
       
      ########## concat all ############

      deriv_dummy_full = tf.concat([deriv_dummy, deriv_dummy_2], axis=1)
      diff_y_dummy_full = tf.concat([diff_y_dummy, diff_y_dummy_2], axis=1)
      diff_x_dummy_full = tf.concat([diff_x_dummy, diff_x_dummy_2], axis=1)
      closest_y_dummy_full = tf.concat([closest_y_dummy, closest_y_dummy_2], axis=1)
      closest_x_dummy_full = tf.concat([closest_x_dummy, closest_x_dummy_2], axis=1)

      return diff_y_dummy_full, diff_x_dummy_full, deriv_dummy_full, closest_x_dummy_full, closest_y_dummy_full

# ## We will need the date information in a numeric version 
# def date_to_numeric(col):
#   datetime = pd.to_datetime(col)
#   return datetime.dt.hour, datetime.dt.day, datetime.dt.month, datetime.dt.year

(12) model/__init__.py: nothing

(13) model/dot_prod.py:
import tensorflow as tf

class DotProductAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
     
  def call(self, queries, keys, values, d_k, mask=None):
    scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))     
    if mask is not None:
      inverse_mask = (mask == False)
      scores += -1e9 * tf.cast(inverse_mask,tf.float32)
    weights = tf.keras.backend.softmax(scores)
     
    #below sets to zero if mask had a row of zeros (softmax would give data leakage)
    if mask is not None:
      weights = tf.math.minimum(tf.math.abs(tf.cast(mask,tf.float32)),tf.math.abs(weights))
     
    return tf.matmul(weights, values)

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, output_shape, projection_shape):
    super().__init__()
    self.attention = DotProductAttention() # Scaled dot product attention
    self.heads = num_heads # Number of attention heads to use
    self.projection_shape = projection_shape # Dimensionality of the linearly projected queries, keys and values
    self.W_q = tf.keras.layers.Dense(projection_shape) # Learned projection matrix for the queries
    self.W_k = tf.keras.layers.Dense(projection_shape) # Learned projection matrix for the keys
    self.W_v = tf.keras.layers.Dense(projection_shape) # Learned projection matrix for the values
    self.W_o = tf.keras.layers.Dense(output_shape) # Learned projection matrix for the multi-head output
    assert projection_shape % self.heads == 0

    #heads must be a factor of projection_shape

  def reshape_tensor(self, x, heads, flag):
    if flag:
      # Tensor shape after reshaping and transposing: (batch_size, seq_length, heads,-1)
      x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))
      x = tf.transpose(x, perm=(0, 2, 1, 3))
    else:
      # Reverting the reshaping and transposing operations: (batch_size, seq_length, projection_shape)
      x = tf.transpose(x, perm=(0, 2, 1, 3))
      x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.projection_shape))
    return x

  def call(self, queries, keys, values, mask=None):
    # Rearrange the queries to be able to compute all heads in parallel
    q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange the keys to be able to compute all heads in parallel
    k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange the values to be able to compute all heads in parallel
    v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Compute the multi-head attention output using the reshaped queries, keys and values
    o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.projection_shape, mask)
    # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

    # Rearrange back the output into concatenated form
    output = self.reshape_tensor(o_reshaped, self.heads, False)
    # Resulting tensor shape: (batch_size, input_seq_length, d_v)

    # Apply one final linear projection to the output to generate the multi-head attention
    # Resulting tensor shape: (batch_size, input_seq_length, d_model)
    return self.W_o(output)

(14) model/losses.py:
import tensorflow as tf
import math as m

def nll(y, μ, log_σ, ϵ=0.001):
  pi = tf.constant(m.pi, dtype=tf.float32)
  y = tf.cast(y, tf.float32)
  mse_per_point = tf.math.square(tf.math.subtract(y, μ))
  lik_per_point = (1 / 2) * tf.math.divide(mse_per_point, tf.math.square(tf.math.exp(log_σ) + ϵ)) + tf.math.log(tf.math.exp(log_σ) + ϵ) + (1/2)*tf.math.log(2*pi)
  sum_lik = tf.math.reduce_sum(lik_per_point)
  sum_mse = tf.math.reduce_sum(mse_per_point)
   
  return lik_per_point, sum_mse, sum_lik, tf.math.reduce_mean(lik_per_point), tf.math.reduce_mean(mse_per_point)

(15) model/taylorformer.py:
import tensorflow as tf
from model import dot_prod

class FFN_1(tf.keras.layers.Layer):
  def __init__(self, output_shape, dropout_rate=0.1):
    super(FFN_1, self).__init__()
    
    self.dense_a = tf.keras.layers.Dense(output_shape)
    self.dense_b = tf.keras.layers.Dense(output_shape)
    self.dense_c = tf.keras.layers.Dense(output_shape)
    self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]     
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, query, training = True):
    ## call layer after first MHA_X
    ## x is the output of MHA_X_1
    ## query is query input to MHA_X_1 

    query = self.dense_a(query)
    x += query
    x = self.layernorm[0](x)
    x_skip = tf.identity(x)
    x = self.dense_b(x)
    x = tf.nn.gelu(x)
    x = self.dropout(x, training=training)
    x = self.dense_c(x)
    x += x_skip
    return self.layernorm[1](x)

class FFN_o(tf.keras.layers.Layer):
  def __init__(self, output_shape, dropout_rate=0.1):
    super(FFN_o, self).__init__()

    self.dense_b = tf.keras.layers.Dense(output_shape)
    self.dense_c = tf.keras.layers.Dense(output_shape)
    self.layernorm = [tf.keras.layers.LayerNormalization() for _ in range(2)]     
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, query, training = True):
   ## query is the output of previous MHA_X layer
   ## x is query input to MHA_X_o 

    x += query
    x = self.layernorm[0](x)
    x_skip = tf.identity(x)
    x = self.dense_b(x)
    x = tf.nn.gelu(x)
    x = self.dropout(x, training=training)
    x = self.dense_c(x)
    x += x_skip
    return self.layernorm[1](x)

class MHA_X_a(tf.keras.layers.Layer):
  def __init__(self,
         num_heads,
         projection_shape,
         output_shape,
         dropout_rate=0.1):
    super(MHA_X_a, self).__init__()
    self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
    self.ffn = FFN_1(output_shape, dropout_rate)

  def call(self, query, key, value, mask, training = True):
    x = self.mha(query, key, value, mask)
    x = self.ffn(x, query, training=training) # Shape `(batch_size, seq_len, output_shape)`.
    return x

class MHA_XY_a(tf.keras.layers.Layer):
  def __init__(self,
         num_heads,
         projection_shape,
         output_shape,
         dropout_rate=0.1):
    super(MHA_XY_a, self).__init__()
    self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
    self.ffn = FFN_1(output_shape, dropout_rate)

  def call(self, query, key, value, mask, training=True):
    x = self.mha(query, key, value, mask)
    x = self.ffn(x, query, training=training) # Shape `(batch_size, seq_len, output_shape)`.

    return x

class MHA_X_b(tf.keras.layers.Layer):
  def __init__(self,
         num_heads,
         projection_shape,
         output_shape,
         dropout_rate=0.1):
    super(MHA_X_b, self).__init__()
    self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
    self.ffn = FFN_o(output_shape, dropout_rate)

  def call(self, query, key, value, mask, training = True):
    x = self.mha(query, key, value, mask)
    x = self.ffn(x, query, training = training) # Shape `(batch_size, seq_len, output_shape)`.

    return x

class MHA_XY_b(tf.keras.layers.Layer):
  def __init__(self,
         num_heads,
         projection_shape,
         output_shape,
         dropout_rate=0.1):
    super(MHA_XY_b, self).__init__()
    self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
    self.ffn = FFN_o(output_shape, dropout_rate)

  def call(self, query, key, value, mask, training=True):
    x = self.mha(query, key, value, mask)
    x = self.ffn(x, query, training=training) # Shape `(batch_size, seq_len, output_shape)`.
    return x

class taylorformer(tf.keras.Model):
  def __init__(self, num_heads,
         projection_shape,
         output_shape,
         num_layers,
         dropout_rate=0.1, target_y_dim=1,
         bound_std = False
         ):
    super().__init__()

    self.num_layers = num_layers
     
    self.mha_x_a = MHA_X_a(num_heads,
         projection_shape,
         output_shape,
         dropout_rate=dropout_rate)
    
    self.mha_x_b = [MHA_X_b(num_heads,
         projection_shape,
         output_shape,
         dropout_rate=dropout_rate) for _ in range(num_layers-1)]

    self.mha_xy_a = MHA_XY_a(num_heads,
         projection_shape,
         output_shape, dropout_rate=dropout_rate)
     
    self.mha_xy_b = [MHA_XY_b(num_heads,
         projection_shape,
         output_shape,
         dropout_rate=dropout_rate) for _ in range(num_layers-1)]

    self.linear_layer = tf.keras.layers.Dense(output_shape)

    self.dense_sigma = tf.keras.layers.Dense(target_y_dim)
    self.dense_last = tf.keras.layers.Dense(target_y_dim)
    self.bound_std = bound_std

  def call(self, input, training=True):
    query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n = input

    x = self.mha_x_a(query_x,query_x, query_x, mask,training=training)
    xy = self.mha_xy_a(query_xy, key_xy, value_xy, mask,training=training)

    for i in range(self.num_layers - 2):

      xy = self.mha_xy_b[i](xy, xy, xy, mask,training=training)
      x = self.mha_x_b[i](x, x, x, mask,training=training)

    xy = self.mha_xy_b[-1](xy, xy, xy, mask,training=training)
    x = self.mha_x_b[-1](x, x, value_x, mask,training=training)

    combo = tf.concat([x,xy], axis = 2)
    z = self.linear_layer(combo)
     
    log_σ = self.dense_sigma(z)

    μ = self.dense_last(z) + y_n

    σ = tf.exp(log_σ)
    if self.bound_std:

      σ = 0.01 + 0.99 * tf.math.softplus(log_σ)

    log_σ = tf.math.log(σ)
   
    return μ, log_σ

(16) model/taylorformer_graph.py:
import tensorflow as tf
from model import losses

def build_graph():
   
  @tf.function(experimental_relax_shapes=True)
  def train_step(taylorformer_model, optimizer, x, y, n_C, n_T, training=True):

    with tf.GradientTape(persistent=True) as tape:

      μ, log_σ = taylorformer_model([x, y, n_C, n_T, training]) 
      _, _, _, likpp, mse = losses.nll(y[:, n_C:n_T+n_C], μ, log_σ)
     
    gradients = tape.gradient(likpp, taylorformer_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, taylorformer_model.trainable_variables))
    return μ, log_σ, likpp, mse

  tf.keras.backend.set_floatx('float32')
  return train_step

(17) model/taylorformer_pipeline.py:
import tensorflow as tf
from tensorflow import keras
import numpy as np
from data_wrangler.feature_extractor import DE, feature_wrapper
from model.taylorformer import taylorformer as taylorformer

class taylorformer_pipeline(keras.models.Model):
   
  def __init__(self, num_heads=4, projection_shape_for_head=4, output_shape=64, rate=0.1, permutation_repeats=1,
         bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=2, MHAX="xxx",**kwargs):
    super().__init__(**kwargs)
    # for testing set permutation_repeats=0
  
    self._permutation_repeats = permutation_repeats
    self.enc_dim = enc_dim
    self.xmin = xmin
    self.xmax = xmax
    self._feature_wrapper = feature_wrapper()
    if MHAX == "xxx":
      self._taylorformer = taylorformer(num_heads=num_heads,dropout_rate=rate,num_layers=num_layers,output_shape=output_shape,
            projection_shape=projection_shape_for_head*num_heads,bound_std=bound_std)
    self._DE = DE()

  def call(self,inputs):

    x, y, n_C, n_T, training = inputs
    #x and y have shape batch size x length x dim

    x = x[:,:n_C+n_T,:]
    y = y[:,:n_C+n_T,:]

    if training == True:   
      x,y = self._feature_wrapper.permute([x, y, n_C, n_T, self._permutation_repeats]) 
     
    x_emb = [self._feature_wrapper.PE([x[:, :, i][:, :, tf.newaxis], self.enc_dim, self.xmin, self.xmax]) for i in range(x.shape[-1])] 
    x_emb = tf.concat(x_emb, axis=-1)

    ######## make mask #######
     
    context_part = tf.concat([tf.ones((n_C,n_C),tf.bool),tf.zeros((n_C,n_T),tf.bool)],axis=-1)
    diagonal_mask = tf.linalg.band_part(tf.ones((n_C+n_T,n_C+n_T),tf.bool),-1,0)
    lower_diagonal_mask = tf.linalg.set_diag(diagonal_mask,tf.zeros(diagonal_mask.shape[0:-1],tf.bool))                                      
    mask = tf.concat([context_part,lower_diagonal_mask[n_C:n_C+n_T,:n_C+n_T]],axis=0) 
     
    ######## create derivative ########

    y_diff, x_diff, d, x_n, y_n = self._DE([y, x, n_C, n_T, training])

    inputs_for_processing = [x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T]

    query_x, key_x, value_x, query_xy, key_xy, value_xy = self._feature_wrapper(inputs_for_processing)
     
    y_n_closest = y_n[:, :, :y.shape[-1]] 

    μ, log_σ = self._taylorformer([query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n_closest],training=training)

    return μ[:, n_C:], log_σ[:, n_C:]
    
def instantiate_taylorformer(dataset,training=True):
  if dataset == "ETT":

    return taylorformer_pipeline(num_heads=6, projection_shape_for_head=11, output_shape=32, rate=0.05, permutation_repeats=0,
         bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1,MHAX="xxx")    

  elif dataset == "exchange":

    return taylorformer_pipeline(num_heads=8, projection_shape_for_head=12, output_shape=32, rate=0.05, permutation_repeats=0,
         bound_std=False, num_layers=3, enc_dim=32, xmin=0.1, xmax=1,MHAX="xxx")
  else:
    print('choose a valid dataset name')          
   
(18) weights_/forecasting/ETT/taylorformer/96/ckpt/check_run_0/checkpoint:
model_checkpoint_path: "ckpt-37"
all_model_checkpoint_paths: "ckpt-37"
all_model_checkpoint_timestamps: 1683890483.0369852
last_preserved_timestamp: 1683889369.1196783

I need you to switch this tensorflow project into a pytorch project. Sure, here is a PyTorch implementation that is equivalent to the provided TensorFlow code.
1. **Data Wrangler**  2. **Pipeline Wrapper**

   
3. **Training Loop**

   ython
import torch
import math as m

def nll(y, μ, log_σ, ϵ=0.001):
    pi = torch.tensor(m.pi)
    y = y.float()
    mse_per_point = torch.square(torch.subtract(y, μ))
    lik_per_point = (1 / 2) * torch.div(mse_per_point, torch.square(torch.exp(log_σ) + ϵ)) + torch.log(torch.exp(log_σ) + ϵ) + (1 / 2) * torch.log(2 * pi)
    sum_lik = torch.sum(lik_per_point)
    sum_mse = torch.sum(mse_per_point)
    
    return lik_per_point, sum_mse, sum_lik, torch.mean(lik_per_point), torch.mean(mse_per_point)
ython
import numpy as np
import pandas as pd
import torch

def dataset_processor(path_to_data):
    # works for exchange and ETTm2 dataset w/o extra features

    pd_array = pd.read_csv(path_to_data)
    data = np.array(pd_array)
    data[:, 0] = np.linspace(-1, 1, data.shape[0])
    # we need to have it between -1 to 1 for each batch item not just overall!!!!!!!!

    data = data.astype("float32")

    training_data = data[:int(0.69 * data.shape[0])]
    val_data = data[int(0.69 * data.shape[0]):int(0.8 * data.shape[0])]
    test_data = data[int(0.8 * data.shape[0]):]

    # scale

    training_data_scaled = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    val_data_scaled = (val_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    test_data_scaled = (test_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)

    x_train, y_train = torch.tensor(training_data_scaled[:, 1:]), torch.tensor(training_data_scaled[:, :1])
    x_val, y_val = torch.tensor(val_data_scaled[:, 1:]), torch.tensor(val_data_scaled[:, :1])
    x_test, y_test = torch.tensor(test_data_scaled[:, 1:]), torch.tensor(test_data_scaled[:, :1])

    return x_train, y_train, x_val, y_val, x_test, y_test

def electricity_processor(path_to_data):
    data = np.load(path_to_data)
    data = (data[322:323].transpose([1, 0])) # shape 14000 x 1

    time = np.linspace(-1, 1, data.shape[0]) # shape 14000
    time = time[:, np.newaxis] # shape 14000 x 1

    data = np.concatenate([time, data], axis=1) # shape 14000 x 2

    training_data = data[:int(0.69 * data.shape[0])]
    val_data = data[int(0.69 * data.shape[0]):int(0.8 * data.shape[0])]
    test_data = data[int(0.8 * data.shape[0]):]

    # scale
    training_data_scaled = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    val_data_scaled = (val_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    test_data_scaled = (test_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)

    x_train, y_train = torch.tensor(training_data_scaled[:, 1:]), torch.tensor(training_data_scaled[:, :1])
    x_val, y_val = torch.tensor(val_data_scaled[:, 1:]), torch.tensor(val_data_scaled[:, :1])
    x_test, y_test = torch.tensor(test_data_scaled[:, 1:]), torch.tensor(test_data_scaled[:, :1])

    return x_train, y_train, x_val, y_val, x_test, y_test

def gp_data_processor(path_to_data_folder):

    x = np.load(path_to_data_folder + "x.npy")
    y = np.load(path_to_data_folder + "y.npy")

    x_train = x[:int(0.99 * x.shape[0])]
    y_train = y[:int(0.99 * y.shape[0])]
    x_val = x[int(0.99 * x.shape[0]):]
    y_val = y[int(0.99 * y.shape[0]):]

    x_test = np.load(path_to_data_folder + "x_test.npy")
    y_test = np.load(path_to_data_folder + "y_test.npy")

    context_n_test = np.load(path_to_data_folder + "context_n_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test, context_n_test

ython
import torch
import numpy as np
from model import dot_prod

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def call(self, x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T):

        dim_x = x_n.shape[-1]
        ##### inputs for the MHA-X head ######
        value_x = y #check if identity is needed
        x_prime = torch.concat([x_emb, x_diff, x_n], dim=2) ### check what is happening with embedding
        query_x = x_prime
        key_x = x_prime

        ##### inputs for the MHA-XY head ######
        y_prime = torch.concat([y, y_diff, d, y_n], dim=-1)
        batch_s = torch.shape(y_prime)[0]
        key_xy_label = torch.zeros((batch_s, n_C + n_T, 1))
        value_xy = torch.concat([y_prime, key_xy_label, x_prime], dim=-1)
        key_xy = value_xy
        
        query_xy_label = torch.concat([torch.zeros((batch_s, n_C, 1)), torch.ones((batch_s, n_T, 1))], dim=1)
        y_prime_masked = torch.concat([self.mask_target_pt([y, n_C, n_T]), self.mask_target_pt([y_diff, n_C, n_T]), self.mask_target_pt([d, n_C, n_T]), y_n], dim=2)

        query_xy = torch.concat([y_prime_masked, query_xy_label, x_prime], dim=-1)

        return query_x, key_x, value_x, query_xy, key_xy, value_xy

    def mask_target_pt(self, inputs):
        y, n_C, n_T = inputs
        dim = y.shape[-1]
        batch_s = y.shape[0]

        mask_y = torch.concat([y[:, :n_C], torch.zeros((batch_s, n_T, dim))], dim=1)
        return mask_y

    def permute(self, inputs):

        x, y, n_C, _, num_permutation_repeats = inputs

        if (num_permutation_repeats < 1):
            return x, y

        else:
            # Shuffle traget only. tf.random.shuffle only works on the first dimension so we need tf.transpose.
            x_permuted = torch.concat([torch.concat([x[:, :n_C, :], torch.transpose(torch.randperm(torch.transpose(x[:, n_C:, :], perm=[1, 0, 2])), perm=[1, 0, 2])], dim=1) for j in range(num_permutation_repeats)], dim=0)

            y_permuted = torch.concat([torch.concat([y[:, :n_C, :], torch.transpose(torch.randperm(torch.transpose(y[:, n_C:, :], perm=[1, 0, 2])), perm=[1, 0, 2])], dim=1) for j in range(num_permutation_repeats)], dim=0)

            return x_permuted, y_permuted

class DE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm_layer = torch.nn.BatchNorm1d()

    def call(self, y, x, n_C, n_T, training):

        if (x.shape[-1] == 1):
            y_diff, x_diff, d, x_n, y_n = self.derivative_function([y, x, n_C, n_T])
        else:
            y_diff, x_diff, d, x_n, y_n = self.derivative_function_2d([y, x, n_C, n_T])

        d_1 = torch.where(torch.isnan(d), torch.tensor(10000.), d)
        d_2 = torch.where(torch.abs(d) > 200., torch.tensor(0.), d)
        d = self.batch_norm_layer(d_2, training=training)

        d_label = torch.where(torch.eq(d_2, d_1), torch.tensor(1.), torch.tensor(0.))
        d = torch.concat([d, d_label], dim=-1)

        return y_diff, x_diff, d, x_n, y_n

###### i think here what we do is calculate the derivative at the given y value and add that in as a feature. This is masked when making predictions
    # so the derivative of other y values are what are seen
    # Based on taylor expansion, a better feature would be including the derivative of the closest x point, where only seen y values are used for the differencing.
    #this derivative wouldn't need masking.

    # #### but you need mutli-dimensional y for the derivative

    ####### and explain why we do this

    def derivative_function(self, inputs):

        y_values, x_values, context_n, target_m = inputs

        epsilon = 0.000002

        batch_size = y_values.shape[0]

        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        #context section

        current_x = x_values[:, :context_n]
        current_y = y_values[:, :context_n]

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_temp, dim=1), context_n, dim=1)

        ix = torch.argsort(torch.norm((current_x - x_temp), dim=-1), dim=-1)[:, :, 1]
        selection_indices = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * context_n), 1), (-1, 1)), torch.reshape(ix, (-1, 1))], dim=1)

        x_closest = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, context_n, dim_x)), selection_indices, dim=1), (batch_size, context_n, dim_x))

        y_closest = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, context_n, dim_y)), selection_indices, dim=1), (batch_size, context_n, dim_y))

        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest

        deriv = y_rep / (epsilon + torch.norm(x_rep, dim=-1, keepdim=True))

        dydx_dummy = deriv

        diff_y_dummy = y_rep

        diff_x_dummy = x_rep

        closest_y_dummy = y_closest

        closest_x_dummy = x_closest

        #target selection

        current_x = x_values[:, context_n:context_n+target_m]
        current_y = y_values[:, context_n:context_n+target_m]

        x_temp = torch.repeat_interleave(torch.unsqueeze(x_values[:, :target_m+context_n], dim=1), target_m, dim=1)
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_values[:, :target_m+context_n], dim=1), target_m, dim=1)

        x_mask = torch.tril(torch.ones((target_m, context_n + target_m), dtype=bool), -1)
        x_mask_inv = (x_mask == False)
        x_mask_float = torch.where(x_mask_inv, torch.tensor(1.), torch.tensor(0.)) * 1000
        x_mask_float_repeat = torch.repeat_interleave(torch.unsqueeze(x_mask_float, dim=0), batch_size, dim=0)
        ix = torch.argsort(torch.norm((current_x - x_temp), dim=-1) + x_mask_float_repeat, dim=-1)[:, :, 1]

        selection_indices = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * target_m), 1), (-1, 1)), torch.reshape(ix, (-1, 1))], dim=1)

        x_closest = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices, dim=1), (batch_size, target_m, dim_x))

        y_closest = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices, dim=1), (batch_size, target_m, dim_y))

        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest

        deriv = y_rep / (epsilon + torch.norm(x_rep, dim=-1, keepdim=True))

        dydx_dummy = torch.cat([dydx_dummy, deriv], dim=1)
        diff_y_dummy = torch.cat([diff_y_dummy, y_rep], dim=1)
        diff_x_dummy = torch.cat([diff_x_dummy, x_rep], dim=1)
        closest_y_dummy = torch.cat([closest_y_dummy, y_closest], dim=1)
        closest_x_dummy = torch.cat([closest_x_dummy, x_closest], dim=1)

        return diff_y_dummy, diff_x_dummy, dydx_dummy, closest_x_dummy, closest_y_dummy

    def derivative_function_2d(self, inputs):

        epsilon = 0.0000

        def dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2):
            #"z" is the second dim of x input
            numerator = y_closest_2 - current_y[:, :, 0] - ((x_closest_2[:, :, :1]-current_x[:, :, 0, :1])*(y_closest_1-current_y[:, :, 0] ))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1] +epsilon)
            denom = x_closest_2[:, :, 1:2] - current_x[:, :, 0, 1:2] - (x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2])*(x_closest_2[:, :, :1]-current_x[:, :, 0, :1])/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
            dydz_pred = numerator/(denom+epsilon)
            return dydz_pred

        def dydx(dydz, current_y, y_closest_1, current_x, x_closest_1):
            dydx = (y_closest_1-current_y[:, :, 0] - dydz*(x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2]))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
            return dydx

        y_values, x_values, context_n, target_m = inputs

        batch_size, length = y_values.shape[0], context_n + target_m

        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        #context section

        current_x = x_values[:, :context_n]
        current_y = y_values[:, :context_n]

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_temp, dim=1), context_n, dim=1)

        ix_1 = torch.argsort(torch.norm((current_x - x_temp), dim=-1), dim=-1)[:, :, 1]
        selection_indices_1 = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * context_n), 1), (-1, 1)), torch.reshape(ix_1, (-1, 1))], dim=1)

        ix_2 = torch.argsort(torch.norm((current_x - x_temp), dim=-1), dim=-1)[:, :, 2]
        selection_indices_2 = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * context_n), 1), (-1, 1)), torch.reshape(ix_2, (-1, 1))], dim=1)

        x_closest_1 = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_1, dim=1), (batch_size, context_n, dim_x)) + torch.randn((batch_size, context_n, dim_x)) * 0.01

        x_closest_2 = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, context_n, dim_x)), selection_indices_2, dim=1), (batch_size, context_n, dim_x)) + torch.randn((batch_size, context_n, dim_x)) * 0.01

        y_closest_1 = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_1, dim=1), (batch_size, context_n, dim_y))

        y_closest_2 = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, context_n, dim_y)), selection_indices_2, dim=1), (batch_size, context_n, dim_y))

        x_rep_1 = current_x[:, :, 0] - x_closest_1
        x_rep_2 = current_x[:, :, 0] - x_closest_2

        y_rep_1 = current_y[:, :, 0] - y_closest_1
        y_rep_2 = current_y[:, :, 0] - y_closest_2

        dydx_2 = dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2)
        dydx_1 = dydx(dydx_2, current_y, y_closest_1, current_x, x_closest_1)

        deriv_dummy = torch.cat([dydx_1, dydx_2], dim=-1)

        diff_y_dummy = torch.cat([y_rep_1, y_rep_2], dim=-1)

        diff_x_dummy = torch.cat([x_rep_1, x_rep_2], dim=-1)

        closest_y_dummy = torch.cat([y_closest_1, y_closest_2], dim=-1)
        closest_x_dummy = torch.cat([x_closest_1, x_closest_2], dim=-1)

        #target selection

        current_x = x_values[:, context_n:context_n+target_m]
        current_y = y_values[:, context_n:context_ ython
import torch
import numpy as np
import pandas as pd

def training_loop(model_pipeline, optimizer, train_x, train_y, test_x, test_y, device, epochs=100, lr_scheduler=None, save_info=False,
                save_name=None):
    model_pipeline.train()
    model_pipeline.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    best_loss = 100
    step = 1
    for epoch in range(epochs):
        # Get predictions from the model
        x, y = train_x, train_y
        # x = np.repeat(np.linspace(-1, 1, n_C + n_T)[np.newaxis, :, np.newaxis], axis=0, repeats=batch_size)  # it doesnt matter what the time is, just the relation between the times.
        optimizer.zero_grad()

        μ, log_σ = model_pipeline([x, y])

        # evaluate loss
        lik_per_point, sum_mse, sum_lik, mean_lik_per_point, mean_mse = nll(y[:, n_C:], μ, log_σ)
        loss = -1. * sum_lik

        # perform back propagation
        loss.backward()

        # perform optimization step
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % 10 == 0:
            model_pipeline.eval()
            with torch.no_grad():
                y_hat, log_σ = model_pipeline([test_x, test_y])
                lik_per_point, sum_mse, sum_lik, mean_lik_per_point, mean_mse = nll(test_y[:, n_C:], y_hat, log_σ)
            val_loss = -1. * sum_lik

            print(f'Epoch: {epoch}, Loss: {-1. * mean_lik_per_point.item():.4f}, Val Loss: {-1. * mean_lik_per_point.item():.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                step = epoch
                if save_info:
                    torch.save(model_pipeline.state_dict(), f'{save_name}_step{step}.ckpt')

def training_loop_classification(model_pipeline, optimizer, train_x, train_y, test_x, test_y, device, epochs=100, lr_scheduler=None, save_info=False,
                save_name=None):
    model_pipeline.train()
    model_pipeline.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    best_loss = 100
    step = 1
    for epoch in range(epochs):
        # Get predictions from the model
        x, y = train_x, train_y
        # x = np.repeat(np.linspace(-1, 1, n_C + n_T)[np.newaxis, :, np.newaxis], axis=0, repeats=batch_size)  # it doesnt matter what the time is, just the relation between the times.
        optimizer.zero_grad()

        logits = model_pipeline([x, y])

        # evaluate loss
        loss = torch.nn.BCEWithLogitsLoss()(logits, y)

        # perform back propagation
        loss.backward()

        # perform optimization step
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % 10 == 0:
            model_pipeline.eval()
            with torch.no_grad():
                logits = model_pipeline([test_x, test_y])
                val_loss = torch.nn.BCEWithLogitsLoss()(logits, y)

            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                step = epoch
                if save_info:
                    torch.save(model_pipeline.state_dict(), f'{save_name}_step{step}.ckpt')

ython
import torch
from torch import nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(self, output_shape, dropout_rate=0.1):
        super().__init__()

        self.dense_a = nn.Linear(output_shape, output_shape)
        self.dense_b = nn.Linear(output_shape, output_shape)
        self.dense_c = nn.Linear(output_shape, output_shape)
        self.layernorm = [nn.LayerNorm(output_shape) for _ in range(2)]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, query, training=True):
        ## call layer after first MHA_X
        ## x is the output of MHA_X_1
        ## query is query input to MHA_X_1

        query = self.dense_a(query)
        x += query
        x = self.layernorm[0](x)
        x_skip = x
        x = self.dense_b(x)
        x = F.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense_c(x)
        x += x_skip
        return self.layernorm[1](x)


class FFN_o(nn.Module):
    def __init__(self, output_shape, dropout_rate=0.1):
        super().__init__()

        self.dense_b = nn.Linear(output_shape, output_shape)
        self.dense_c = nn.Linear(output_shape, output_shape)
        self.layernorm = [nn.LayerNorm(output_shape) for _ in range(2)]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, query, training=True):
        ## query is the output of previous MHA_X layer
        ## x is query input to MHA_X_o

        x += query
        x = self.layernorm[0](x)
        x_skip = x
        x = self.dense_b(x)
        x = F.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense_c(x)
        x += x_skip
        return self.layernorm[1](x)


class MHA_X_a(nn.Module):
    def __init__(self,
                 num_heads,
                 projection_shape,
                 output_shape,
                 dropout_rate=0.1):
        super().__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.
        return x


class MHA_XY_a(nn.Module):
    def __init__(self,
                 num_heads,
                 projection_shape,
                 output_shape,
                 dropout_rate=0.1):
        super().__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.

        return x


class MHA_X_b(nn.Module):
    def __init__(self,
                 num_heads,
                 projection_shape,
                 output_shape,
                 dropout_rate=0.1):
        super().__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.

        return x


class MHA_XY_b(nn.Module):
    def __init__(self,
                 num_heads,
                 projection_shape,
                 output_shape,
                 dropout_rate=0.1):
        super().__init__()
        self.mha = dot_prod.MultiHeadAttention(num_heads, output_shape, projection_shape)
        self.ffn = FFN_o(output_shape, dropout_rate)

    def forward(self, query, key, value, mask, training=True):
        x = self.mha(query, key, value, mask)
        x = self.ffn(x, query, training=training)  # Shape `(batch_size, seq_len, output_shape)`.
        return x


class taylorformer(nn.Module):
    def __init__(self, num_heads,
                 projection_shape,
                 output_shape,
                 num_layers,
                 dropout_rate=0.1, target_y_dim=1,
                 bound_std=False
                 ):
        super().__init__()

        self.num_layers = num_layers

        self.mha_x_a = MHA_X_a(num_heads,
                               projection_shape,
                               output_shape,
                               dropout_rate=dropout_rate)

        self.mha_x_b = [MHA_X_b(num_heads,
                               projection_shape,
                               output_shape,
                               dropout_rate=dropout_rate) for _ in range(num_layers - 1)]

        self.mha_xy_a = MHA_XY_a(num_heads,
                                 projection_shape,
                                 output_shape, dropout_rate=dropout_rate)

        self.mha_xy_b = [MHA_XY_b(num_heads,
                                 projection_shape,
                                 output_shape, dropout_rate=dropout_rate) for _ in range(num_layers - 1)]

        self.linear_layer = nn.Linear(output_shape, output_shape)

        self.dense_sigma = nn.Linear(output_shape, target_y_dim)
        self.dense_last = nn.Linear(output_shape, target_y_dim)
        self.bound_std = bound_std

    def forward(self, input, training=True):
        query_x, key_x, value_x, query_xy, key_xy, value_xy, mask, y_n = input

        x = self.mha_x_a(query_x, query_x, query_x, mask, training=training)
        xy = self.mha_xy_a(query_xy, key_xy, value_xy, mask, training=training)

        for i in range(self.num_layers - 2):

            xy = self.mha_xy_b[i](xy, xy, xy, mask, training=training)
            x = self.mha_x_b[i](x, x, x, mask, training=training)

        xy = self.mha_xy_b[-1](xy, xy, xy, mask, training=training)
        x = self.mha_x_b[-1](x, x, value_x, mask, training=training)

        combo = torch.cat([x, xy], axis=2)
        z = self.linear_layer(combo)

        log_sigma = self.dense_sigma(z)

        mu = self.dense_last(z) + y_n

        sigma = torch.exp(log_sigma)
        if self.bound_std:

            sigma = 0.01 + 0.99 * torch.softplus(log_sigma)

        log_sigma = torch.log(sigma)

        return mu, log_sigma
ython
model = taylorformer(num_heads=6, projection_shape_for_head=8, output_shape=48, rate=0.1, permutation_repeats=0,
                    bound_std=False, num_layers=4, enc_dim=32, xmin=0.1, xmax=1, MHAX="xxx").cuda()
ython
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
training_loop(model, optimizer, train_x, train_y, test_x, test_y, device="cuda", epochs=100, lr_scheduler=lr_scheduler, save_info=True,
                save_name="taylorformer")
﻿
Sure, here is the PyTorch implementation of the training loop and data pre-processing functions:﻿
python
import torch
import numpy as np
from model import dot_prod, taylorformer, FFN, MHA_X_a, MHA_XY_a, MHA_X_b, MHA_XY_b, FFN_o
from data_wrangler import dataset_processor, electricity_processor, gp_data_processor, FeatureExtractor, DE


def training_loop(model_pipeline, optimizer, train_x, train_y, test_x, test_y, device, epochs=100, lr_scheduler=None, save_info=False,
                save_name=None):
    model_pipeline.train()
    model_pipeline.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    best_loss = 100
    step = 1
    for epoch in range(epochs):
        # Get predictions from the model
        x, y = train_x, train_y
        # x = np.repeat(np.linspace(-1, 1, n_C + n_T)[np.newaxis, :, np.newaxis], axis=0, repeats=batch_size)  # it doesnt matter what the time is, just the relation between the times.
        optimizer.zero_grad()

        μ, log_σ = model_pipeline([x, y])

        # evaluate loss
        lik_per_point, sum_mse, sum_lik, mean_lik_per_point, mean_mse = nll(y[:, n_C:], μ, log_σ)
        loss = -1. * sum_lik

        # perform back propagation
        loss.backward()

        # perform optimization step
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % 10 == 0:
            model_pipeline.eval()
            with torch.no_grad():
                y_hat, log_σ = model_pipeline([test_x, test_y])
                lik_per_point, sum_mse, sum_lik, mean_lik_per_point, mean_mse = nll(test_y[:, n_C:], y_hat, log_σ)
            val_loss = -1. * sum_lik

            print(f'Epoch: {epoch}, Loss: {-1. * mean_lik_per_point.item():.4f}, Val Loss: {-1. * mean_lik_per_point.item():.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                step = epoch
                if save_info:
                    torch.save(model_pipeline.state_dict(), f'{save_name}_step{step}.ckpt')


def training_loop_classification(model_pipeline, optimizer, train_x, train_y, test_x, test_y, device, epochs=100, lr_scheduler=None, save_info=False,
                save_name=None):
    model_pipeline.train()
    model_pipeline.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    best_loss = 100
    step = 1
    for epoch in range(epochs):
        # Get predictions from the model
        x, y = train_x, train_y
        # x = np.repeat(np.linspace(-1, 1, n_C + n_T)[np.newaxis, :, np.newaxis], axis=0, repeats=batch_size)  # it doesnt matter what the time is, just the relation between the times.
        optimizer.zero_grad()

        logits = model_pipeline([x, y])

        # evaluate loss
        loss = torch.nn.BCEWithLogitsLoss()(logits, y)

        # perform back propagation
        loss.backward()

        # perform optimization step
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % 10 == 0:
            model_pipeline.eval()
            with torch.no_grad():
                logits = model_pipeline([test_x, test_y])
                val_loss = torch.nn.BCEWithLogitsLoss()(logits, y)

            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                step = epoch
                if save_info:
                    torch.save(model_pipeline.state_dict(), f'{save_name}_step{step}.ckpt')


def dataset_processor(path_to_data):
    # works for exchange and ETTm2 dataset w/o extra features

    pd_array = pd.read_csv(path_to_data)
    data = np.array(pd_array)
    data[:, 0] = np.linspace(-1, 1, data.shape[0])
    # we need to have it between -1 to 1 for each batch item not just overall!!!!!!!!

    data = data.astype("float32")

    training_data = data[:int(0.69 * data.shape[0])]
    val_data = data[int(0.69 * data.shape[0]):int(0.8 * data.shape[0])]
    test_data = data[int(0.8 * data.shape[0]):]

    # scale

    training_data_scaled = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    val_data_scaled = (val_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    test_data_scaled = (test_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)

    train_x, train_y = torch.tensor(training_data_scaled[:, 1:]), torch.tensor(training_data_scaled[:, :1])
    test_x, test_y = torch.tensor(val_data_scaled[:, 1:]), torch.tensor(val_data_scaled[:, :1])
    val_x, val_y = torch.tensor(test_data_scaled[:, 1:]), torch.tensor(test_data_scaled[:, :1])

    return train_x, train_y, val_x, val_y, test_x, test_y


def electricity_processor(path_to_data):
    data = np.load(path_to_data)
    data = (data[322:323].transpose([1, 0])) # shape 14000 x 1

    time = np.linspace(-1, 1, data.shape[0]) # shape 14000
    time = time[:, np.newaxis] # shape 14000 x 1

    data = np.concatenate([time, data], axis=1) # shape 14000 x 2

    training_data = data[:int(0.69 * data.shape[0])]
    val_data = data[int(0.69 * data.shape[0]):int(0.8 * data.shape[0])]
    test_data = data[int(0.8 * data.shape[0]):]

    # scale
    training_data_scaled = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    val_data_scaled = (val_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
    test_data_scaled = (test_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)

    train_x, train_y = torch.tensor(training_data_scaled[:, 1:]), torch.tensor(training_data_scaled[:, :1])
    test_x, test_y = torch.tensor(val_data_scaled[:, 1:]), torch.tensor(val_data_scaled[:, :1])
    val_x, val_y = torch.tensor(test_data_scaled[:, 1:]), torch.tensor(test_data_scaled[:, :1])

    return train_x, train_y, val_x, val_y, test_x, test_y


def gp_data_processor(path_to_data_folder):

    x = np.load(path_to_data_folder + "x.npy")
    y = np.load(path_to_data_folder + "y.npy")

    train_x = x[:int(0.99 * x.shape[0])]
    train_y = y[:int(0.99 * y.shape[0])]
    val_x = x[int(0.99 * x.shape[0]):]
    val_y = y[int(0.99 * y.shape[0]):]

    test_x = np.load(path_to_data_folder + "x_test.npy")
    test_y = np.load(path_to_data_folder + "y_test.npy")

    context_n_test = np.load(path_to_data_folder + "context_n_test.npy")

    return train_x, train_y, val_x, val_y, test_x, test_y, context_n_test


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def call(self, x_emb, y, y_diff, x_diff, d, x_n, y_n, n_C, n_T):

        dim_x = x_n.shape[-1]
        ##### inputs for the MHA-X head ######
        value_x = y #check if identity is needed
        x_prime = torch.cat([x_emb, x_diff, x_n], dim=2) ### check what is happening with embedding
        query_x = x_prime
        key_x = x_prime

        ##### inputs for the MHA-XY head ######
        y_prime = torch.cat([y, y_diff, d, y_n], dim=-1)
        batch_s = torch.shape(y_prime)[0]
        key_xy_label = torch.zeros((batch_s, n_C + n_T, 1))
        value_xy = torch.cat([y_prime, key_xy_label, x_prime], dim=-1)
        key_xy = value_xy
        
        query_xy_label = torch.concat([torch.zeros((batch_s, n_C, 1)), torch.ones((batch_s, n_T, 1))], dim=1)
        y_prime_masked = torch.concat([self.mask_target_pt([y, n_C, n_T]), self.mask_target_pt([y_diff, n_C, n_T]), self.mask_target_pt([d, n_C, n_T]), y_n], dim=2)

        query_xy = torch.concat([y_prime_masked, query_xy_label, x_prime], dim=-1)

        return query_x, key_x, value_x, query_xy, key_xy, value_xy

    def mask_target_pt(self, inputs):
        y, n_C, n_T = inputs
        dim = y.shape[-1]
        batch_s = y.shape[0]

        mask_y = torch.concat([y[:, :n_C], torch.zeros((batch_s, n_T, dim))], dim=1)
        return mask_y

    def permute(self, inputs):

        x, y, n_C, _, num_permutation_repeats = inputs

        if (num_permutation_repeats < 1):
            return x, y

        else:
            # Shuffle traget only. tf.random.shuffle only works on the first dimension so we need tf.transpose.
            x_permuted = torch.concat([torch.concat([x[:, :n_C, :], torch.transpose(torch.randperm(torch.transpose(x[:, n_C:, :], perm=[1, 0, 2])), perm=[1, 0, 2])], dim=1) for j in range(num_permutation_repeats)], dim=0)

            y_permuted = torch.concat([torch.concat([y[:, :n_C, :], torch.transpose(torch.randperm(torch.transpose(y[:, n_C:, :], perm=[1, 0, 2])), perm=[1, 0, 2])], dim=1) for j in range(num_permutation_repeats)], dim=0)

            return x_permuted, y_permuted


class DE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm_layer = torch.nn.BatchNorm1d()

    def call(self, y, x, n_C, n_T, training):

        if (x.shape[-1] == 1):
            y_diff, x_diff, d, x_n, y_n = self.derivative_function([y, x, n_C, n_T])
        else:
            y_diff, x_diff, d, x_n, y_n = self.derivative_function_2d([y, x, n_C, n_T])

        d_1 = torch.where(torch.isnan(d), torch.tensor(10000.), d)
        d_2 = torch.where(torch.abs(d) > 200., torch.tensor(0.), d)
        d = self.batch_norm_layer(d_2, training=training)

        d_label = torch.where(torch.eq(d_2, d_1), torch.tensor(1.), torch.tensor(0.))
        d = torch.cat([d, d_label], dim=-1)

        return y_diff, x_diff, d, x_n, y_n

###### i think here what we do is calculate the derivative at the given y value and add that in as a feature. This is masked when making predictions
    # so the derivative of other y values are what are seen
    # Based on taylor expansion, a better feature would be including the derivative of the closest x point, where only seen y values are used for the differencing.
    #this derivative wouldn't need masking.

    # #### but you need mutli-dimensional y for the derivative

    ####### and explain why we do this

    def derivative_function(self, inputs):

        y_values, x_values, context_n, target_m = inputs

        epsilon = 0.000002

        batch_size = y_values.shape[0]

        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        #context section

        current_x = x_values[:, :context_n]
        current_y = y_values[:, :context_n]

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_temp, dim=1), context_n, dim=1)

        ix = torch.argsort(torch.norm((current_x - x_temp), dim=-1), dim=-1)[:, :, 1]
        selection_indices = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * context_n), 1), (-1, 1)), torch.reshape(ix, (-1, 1))], dim=1)

        x_closest = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, context_n, dim_x)), selection_indices, dim=1), (batch_size, context_n, dim_x))

        y_closest = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, context_n, dim_y)), selection_indices, dim=1), (batch_size, context_n, dim_y))

        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest

        deriv = y_rep / (epsilon + torch.norm(x_rep, dim=-1, keepdim=True))

        dydx_dummy = deriv

        diff_y_dummy = y_rep

        diff_x_dummy = x_rep

        closest_y_dummy = y_closest

        closest_x_dummy = x_closest

        #target selection

        current_x = x_values[:, context_n:context_n+target_m]
        current_y = y_values[:, context_n:context_n+target_m]

        x_temp = torch.repeat_interleave(torch.unsqueeze(x_values[:, :target_m+context_n], dim=1), target_m, dim=1)
        y_temp = torch.repeat_interleave(torch.unsqueeze(y_values[:, :target_m+context_n], dim=1), target_m, dim=1)

        x_mask = torch.tril(torch.ones((target_m, context_n + target_m), dtype=bool), -1)
        x_mask_inv = (x_mask == False)
        x_mask_float = torch.where(x_mask_inv, torch.tensor(1.), torch.tensor(0.)) * 1000
        x_mask_float_repeat = torch.repeat_interleave(torch.unsqueeze(x_mask_float, dim=0), batch_size, dim=0)
        ix = torch.argsort(torch.norm((current_x - x_temp), dim=-1) + x_mask_float_repeat, dim=-1)[:, :, 1]

        selection_indices = torch.cat([torch.reshape(torch.repeat_interleave(torch.arange(batch_size * target_m), 1), (-1, 1)), torch.reshape(ix, (-1, 1))], dim=1)

        x_closest = torch.reshape(torch.gather(torch.reshape(x_temp, (-1, target_m+context_n, dim_x)), selection_indices, dim=1), (batch_size, target_m, dim_x))

        y_closest = torch.reshape(torch.gather(torch.reshape(y_temp, (-1, target_m+context_n, dim_y)), selection_indices, dim=1), (batch_size, target_m, dim_y))

        x_rep = current_x[:, :, 0] - x_closest
        y_rep = current_y[:, :, 0] - y_closest

        deriv = y_rep / (epsilon + torch.norm(x_rep, dim=-1, keepdim=True))

        dydx_dummy = torch.cat([dydx_dummy, deriv], dim=1)
        diff_y_dummy = torch.cat([diff_y_dummy, y_rep], dim=1)
        diff_x_dummy = torch.cat([diff_x_dummy, x_rep], dim=1)
        closest_y_dummy = torch.cat([closest_y_dummy, y_closest], dim=1)
        closest_x_dummy = torch.cat([closest_x_dummy, x_closest], dim=1)

        return diff_y_dummy, diff_x_dummy, dydx_dummy, closest_x_dummy, closest_y_dummy

    def derivative_function_2d(self, inputs):

        epsilon = 0.0000

        def dydz(current_y, y_closest_1, y_closest_2, current_x, x_closest_1, x_closest_2):
            #"z" is the second dim of x input
            numerator = y_closest_2 - current_y[:, :, 0] - ((x_closest_2[:, :, :1]-current_x[:, :, 0, :1])*(y_closest_1-current_y[:, :, 0] ))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1] +epsilon)
            denom = x_closest_2[:, :, 1:2] - current_x[:, :, 0, 1:2] - (x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2])*(x_closest_2[:, :, :1]-current_x[:, :, 0, :1])/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
            dydz_pred = numerator/(denom+epsilon)
            return dydz_pred

        def dydx(dydz, current_y, y_closest_1, current_x, x_closest_1):
            dydx = (y_closest_1-current_y[:, :, 0] - dydz*(x_closest_1[:, :, 1:2]-current_x[:, :, 0, 1:2]))/(x_closest_1[:, :, :1]-current_x[:, :, 0, :1]+epsilon)
            return dydx

        y_values, x_values, context_n, target_m = inputs

        batch_size, length = y_values.shape[0], context_n + target_m

        dim_x = x_values.shape[-1]
        dim_y = y_values.shape[-1]

        #context section

        current_x = x_values[:, :context_n]
        current_y = y_values[:, :context_n]

        x_temp = x_values[:, :context_n]
        x_temp = torch.repeat_interleave(torch.unsqueeze(x_temp, dim=1), context_n, dim=1)

        y_temp = y_values[:, :context_n]
        y
