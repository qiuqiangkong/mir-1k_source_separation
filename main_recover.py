'''
SUMMARY:  recover wav from separated spectrogram
AUTHOR:   Qiuqiang Kong
Created:  2016.05.16
Modified: 2016.08.05
          2016.08.11 update
--------------------------------------
'''
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from mir_eval.separation import bss_eval_sources
import pickle
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.models import *
from hat.layers.core import *
from hat.layers.cnn import *
from hat.layers.pool import *
from hat.callbacks import *
from hat import serializations
from hat.preprocessing import mat_2d_to_3d
import hat.backend as K
import config as cfg
import os
import prepare_data as pp_data
import matplotlib.pyplot as plt
from main_dnn import mul


# hyper params
n_time=11
md = serializations.load( cfg.md_fd+'/md10.p' )
fe_fd = cfg.fe_fft_fd

# all songs
names = os.listdir( cfg.wav_fd )
names = sorted( names )
pure_names = [ name[0:-4] for name in names ]

# train phase or test phase
tr_phase = True


# recover for all songs
for na in names:
    if na.endswith('.wav') and not na.endswith('lyrics.wav') and not na.endswith('lyric.wav'):
        if len(na.split('_'))==3:
            gt_mix, gt_left, gt_right = pp_data.GetSingleSongData( fe_fd, na[0:-4], tr_phase=tr_phase )
            if gt_mix is not None:
                print na

                # ground truth, shape: (batch_num, n_time, n_freq)
                gt_mix_3d = mat_2d_to_3d( gt_mix, n_time, n_time )
                gt_left_3d = mat_2d_to_3d( gt_left, n_time, n_time )
                gt_right_3d = mat_2d_to_3d( gt_right, n_time, n_time )
        
                # predict
                [pred_left_3d, pred_right_3d] = md.predict( np.abs(gt_mix_3d) )
                
                # recover 3d to spectrogram
                pred_left = np.concatenate( pred_left_3d, axis=0 )
                pred_right = np.concatenate( pred_right_3d, axis=0 )
            
                # recover wav
                s_pred_left = pp_data.recover_wav( pred_left, gt_mix )
                s_pred_right = pp_data.recover_wav( pred_right, gt_mix )
                s_gt_left = pp_data.recover_gt_wav( gt_left )
                s_gt_right = pp_data.recover_gt_wav( gt_right )
                s_gt_mix = pp_data.recover_gt_wav( gt_mix )
                
                # evaluate
                sdr, sir, sar, perm = bss_eval_sources( s_gt_left, s_pred_left )
                print sdr, sir, sar, perm
                sdr, sir, sar, perm = bss_eval_sources( s_gt_right, s_pred_right )
                print sdr, sir, sar, perm
                
                # write out wavs
                pp_data.write_wav( s_gt_mix, 16000., cfg.results_fd + '/' + na + '_gt_mix.wav' )
                pp_data.write_wav( s_pred_left, 16000., cfg.results_fd + '/' + na + '_pred_left.wav' )
                pp_data.write_wav( s_pred_right, 16000., cfg.results_fd + '/' + na + '_pred_right.wav' )

                pause