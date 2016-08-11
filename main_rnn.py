'''
SUMMARY:  do separation and save out separated spectrogram
          After 100 epochs, abjones_1_01.wav: SDR_left=2.62, SDR_right=4.85
          3.4 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.05.16
Modified: 2016.07.29 Use 1-frame DNN
          2016.08.05 update
          2016.08.11 update
--------------------------------------
'''
import numpy as np
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import os
from Hat.models import *
from Hat.layers.rnn import *
from Hat.layers.core import *
from Hat.layers.cnn import *
from Hat.layers.pool import *
from Hat.layers.rnn import *
from Hat.callbacks import *
import Hat.backend as K
import prepare_data as pp_data
import config as cfg
import cPickle
from main_dnn import mul

n_time = 11
n_in = 513
n_hid = 500
    
def weighted_norm_l1( in0, mask_pred, y_gt ):
    return K.mean( K.sum( K.abs(in0*mask_pred - y_gt), axis=-1 ) )
    
# loss function
def loss_func( out_nodes, inter_nodes, gt_nodes ):
    [in0, mask_pred1, mask_pred2] = inter_nodes
    [gt1, gt2] = gt_nodes
    return weighted_norm_l1( in0, mask_pred1, gt1 ) + weighted_norm_l1( in0, mask_pred2, gt2 )
    
# lambda function
def mul( x, y ):
    return x * y
    
### train blind source separation
def train_bss():
    
    # DO NOT DELETE! Load data for speed up
    # you can annotate the code below after run the first time. 
    t1 = time.time()
    tr_X, tr_y_left, tr_y_right = pp_data.get_all_3d_data( cfg.fe_fft_fd, n_time, tr_phase=True )
    cPickle.dump( [tr_X, tr_y_left, tr_y_right], open( 'Results/tmp.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
    t2 = time.time()
    print 'loading data time: ', t2-t1
    
    
    # load data
    [tr_X, tr_y_left, tr_y_right] = cPickle.load( open( 'Results/tmp.p', 'rb' ) )
    print tr_X.shape 
    
    in1 = InputLayer( in_shape=( (n_time, n_in) ) )
    a1 = SimpleRnn( n_hid, act='relu' )( in1 )
    a2 = Dropout( 0.2 )( a1 )
    a3 = SimpleRnn( n_hid, act='relu' )( a2 )
    a4 = Dense( n_hid, act='relu' )( a3 )
    a5 = Dropout( 0.2 )( a4 )

    b1 = Dense( n_hid, act='relu' )( a5 )
    b2 = Dropout( 0.2 )( b1 )
    b3 = Dense( n_hid, act='relu' )( b2 )
    b4 = Dropout( 0.2 )( b3 )
    b5 = Dense( n_in, act='sigmoid' )( b4 )
    
    c1 = Dense( n_hid, act='relu' )( a5 )
    c2 = Dropout( 0.2 )( c1 )
    c3 = Dense( n_hid, act='relu' )( c2 )
    c4 = Dropout( 0.2 )( c3 )
    c5 = Dense( n_in, act='sigmoid' )( c4 )
    
    out_b = Lambda( mul )( [b5, in1] )
    out_c = Lambda( mul )( [c5, in1] )
    
    md = Model( in_layers=[in1], out_layers=[out_b, out_c], inter_layers=[in1, b5, c5] )
    md.summary()
    
    # validation
    validation = Validation( tr_x=[np.abs(tr_X)], tr_y=[np.abs(tr_y_left), np.abs(tr_y_right)], \
                 batch_size=100, metrics=[loss_func], call_freq=1, dump_path='validation.p' )

    # save model
    if not os.path.exists('Md'): os.makedirs('Md')
    save_model = SaveModel( dump_fd='Md', call_freq=2 )
    
    # callbacks
    callbacks = [ validation, save_model ]
    
    # optimizer
    optimizer = Adam(1e-4)
    
    # fit model
    md.fit( [np.abs(tr_y_left)+np.abs(tr_y_right)], [np.abs(tr_y_left), np.abs(tr_y_right)], \
        batch_size=100, n_epochs=1001, loss_func=loss_func, optimizer=optimizer, callbacks=callbacks, verbose=1 )

if __name__ == '__main__':
    train_bss()