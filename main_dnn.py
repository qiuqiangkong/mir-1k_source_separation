"""
SUMMARY:  do separation and save out separated spectrogram
          After 100 epochs, abjones_1_01.wav: SDR_left=2.62, SDR_right=4.85
          3.4 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.05.16
Modified: 2017.08.12
--------------------------------------
"""
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import numpy as np
import os
from hat.models import *
from hat.layers.core import *
from hat.layers.cnn import *
from hat.layers.pool import *
from hat.layers.rnn import *
from hat.callbacks import *
import hat.objectives as obj
import hat.backend as K
import prepare_data as pp_data
import config as cfg
import cPickle
import time

# hyper params
n_time = 11
n_in = 513
n_hid = 500
    
# loss function
def loss_func(md):
    [in0, mask_pred1, mask_pred2] = md.any_nodes_
    [gt1, gt2] = md.gt_nodes_
    return obj.norm_lp(in0*mask_pred1, gt1, 2) + obj.norm_lp(in0*mask_pred2, gt2, 2)
    
# lambda function
def mul(inputs, **kwargs):
    return inputs[0] * inputs[1]
    
### train blind source separation
def train_bss():
    
    # DO NOT DELETE! Load data for speed up
    # you can annotate the code below after run the first time. 
    
    t1 = time.time()
    tr_X, tr_y_left, tr_y_right = pp_data.get_all_3d_data(cfg.fe_fft_fd, n_time, tr_phase=True)
    cPickle.dump([tr_X, tr_y_left, tr_y_right], open(cfg.results_fd + '/tmp.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    t2 = time.time()
    print 'loading data time: ', t2-t1
    
    
    # load data
    [tr_X, tr_y_left, tr_y_right] = cPickle.load(open(cfg.results_fd+'/tmp.p', 'rb'))    
    print tr_X.shape 
    
    # build model
    in1 = InputLayer(in_shape=((n_time, n_in)), name='in1')
    a1 = Dense(n_hid, act='relu', name='a1')(in1)
    a2 = Dropout(0.2, name='a2')(a1)
    a3 = Dense(n_hid, act='relu', name='a3')(a2)
    a4 = Dropout(0.2, name='a4')(a3)
    a5 = Dense(n_hid, act='relu', name='a5')(a4)
    a6 = Dropout(0.2, name='a6')(a5)
    b1 = Dense(n_in, act='sigmoid', name='a7')(a6)     # mask_left
    c1 = Dense(n_in, act='sigmoid', name='a8')(a6)     # mask_right
    out_b = Lambda(mul, name='out_b')([b1, in1])
    out_c = Lambda(mul, name='out_c')([c1, in1])
    
    md = Model(in_layers=[in1], out_layers=[out_b, out_c], any_layers=[in1, b1, c1])
    md.compile()
    md.summary()
    
    # validation
    validation = Validation(tr_x=[np.abs(tr_X)], 
                            tr_y=[np.abs(tr_y_left), np.abs(tr_y_right)], 
                            batch_size=100, 
                            metrics=[loss_func], 
                            call_freq=1)

    # save model
    if not os.path.exists(cfg.md_fd): os.makedirs(cfg.md_fd)
    save_model = SaveModel(dump_fd=cfg.md_fd, call_freq=2)
    
    # callbacks
    callbacks = [validation, save_model]
    
    # optimizer
    optimizer = Adam(1e-3)
    
    # fit model
    md.fit(x=[np.abs(tr_X)], 
           y=[np.abs(tr_y_left), np.abs(tr_y_right)], 
           batch_size=100, 
           n_epochs=101, 
           loss_func=loss_func, 
           optimizer=optimizer, 
           callbacks=callbacks, 
           verbose=1)

if __name__ == '__main__':
    train_bss()