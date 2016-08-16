'''
SUMMARY:  Calculate features for left, right, mix respectively
AUTHOR:   Qiuqiang Kong
Created:  2016.05.16
Modified: 2016.08.11 update
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import numpy as np
import os
from scipy import signal
import cPickle
import config as cfg
import wavio
import matplotlib.pyplot as plt
from hat.preprocessing import mat_2d_to_3d


### Extract features
# readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate complex spectrogram
def GetSpectrogram( x ):
    # win = np.hamming(1024)
    win = np.ones(1024)/1024.
    [f, t, X] = signal.spectral.spectrogram( x, window=win, nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='complex' )
    X = X.T		# size: N*(nperseg/2+1)
    return X

# calculate all features
def CalculateAllSpectrogram( fe_fd ):
    names = os.listdir( cfg.wav_fd )
    names = sorted( names )
    cnt = 0
    for na in names:
        if na.endswith( '.wav' ) and not na.endswith('lyrics.wav') and not na.endswith('lyric.wav'):
            if len(na.split('_'))==3:                
                print cnt, na
                path = cfg.wav_fd + '/' + na
                data, fs = readwav( path )
                assert fs==16000
                left = data[:,0]
                right = data[:,1]
                mix = np.mean( data, axis=1 )        
        
                # get left, right, mix complex spectrogram respectively
                X_left = GetSpectrogram( left )
                X_right = GetSpectrogram( right )
                X_mix = GetSpectrogram( mix )
                
                # dump
                cPickle.dump( X_left, open( fe_fd+'/left/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
                cPickle.dump( X_right, open( fe_fd+'/right/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
                cPickle.dump( X_mix, open( fe_fd+'/mix/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
  
                cnt += 1
    print cnt

### Prepare data
def na_in_tr_list( na ):
    for e in cfg.tr_list:
        if na.startswith( e ):
            return True
    return False
        
# get list of features of tr_dataset or te_dataset
def GetListData( fe_fd, tr_phase ):
    mix_names = sorted( os.listdir( fe_fd + '/mix' ) )
    left_names = sorted( os.listdir( fe_fd + '/left' ) )
    right_names = sorted( os.listdir( fe_fd + '/right' ) )

    # mix channel
    X_list = []
    cnt = 0
    for na in mix_names:
        X = cPickle.load( open( fe_fd + '/mix/' + na, 'rb' ) )
        if tr_phase is True and na_in_tr_list( na ) is True:
            X_list.append( X )
        if tr_phase is False and na_in_tr_list( na ) is False:
            X_list.append( X )
        
    
    # left channel
    y_left_list = []
    for na in mix_names:
        y_left = cPickle.load( open( fe_fd + '/left/' + na, 'rb' ) )
        if tr_phase is True and na_in_tr_list( na ) is True:
            y_left_list.append( y_left )
            #break
        if tr_phase is False and na_in_tr_list( na ) is False:
            y_left_list.append( y_left )
        
            
    # right channel
    y_right_list = []
    for na in mix_names:
        y_right = cPickle.load( open( fe_fd + '/right/' + na, 'rb' ) )
        if tr_phase is True and na_in_tr_list( na ) is True:
            y_right_list.append( y_right )
            #break
        if tr_phase is False and na_in_tr_list( na ) is False:
            y_right_list.append( y_right )
        
            
    return X_list, y_left_list, y_right_list
    

# load one song's 2d feature, shape: (N, n_freq)
def GetSingleSongData( fe_fd, na, tr_phase ):
    mix_na = fe_fd + '/mix/' + na + '.p'
    left_na = fe_fd + '/left/' + na + '.p'
    right_na = fe_fd + '/right/' + na + '.p'
    
    if ( tr_phase is True and na_in_tr_list( na ) is True ) or ( tr_phase is False and na_in_tr_list( na ) is False ):
        Xmix = cPickle.load( open( fe_fd + '/mix/' + na + '.p', 'rb' ) )
        Xleft = cPickle.load( open( fe_fd + '/left/' + na + '.p', 'rb' ) )
        Xright = cPickle.load( open( fe_fd + '/right/' + na + '.p', 'rb' ) )    
    else:
        Xmix, Xleft, Xright = None, None, None
        
    return Xmix, Xleft, Xright
   
# list of 2d data to 3d data
def list_to_3d( X_list, n_time ):
    X3d_all = []
    for X in X_list:
        X3d = mat_2d_to_3d( X, n_time, n_time )
        X3d_all.append( X3d )
    X3d_all = np.concatenate( X3d_all, axis=0 )
    return X3d_all
    
# get 3d feature of all songs, shape: (N, agg_num, n_freq)
def get_all_3d_data( fe_fd, n_time, tr_phase ):
    X_list, y_left_list, y_right_list = GetListData( fe_fd, tr_phase )
    X3d = list_to_3d( X_list, n_time )              # shape: (N, agg_num, n_freq)
    y3d_left = list_to_3d( y_left_list, n_time )    # shape: (N, agg_num, n_freq)
    y3d_right = list_to_3d( y_right_list, n_time )  # shape: (N, agg_num, n_freq)
    return X3d, y3d_left, y3d_right
    
    
### Recover wav
# recover pred spectrogram's phase from ground truth's phase
def real_to_complex( val, gt_X ):
    theta = np.angle( gt_X )
    cmplx = val * np.cos( theta ) + val * np.sin( theta ) * 1j
    return cmplx
    
# recover whole spectrogram from half spectrogram
def half_to_whole( X ):
    return np.hstack( ( X, np.fliplr( np.conj( X[:,1:-1] ) ) ) )

# recover wav from whole spectrogram
def ifft_to_wav( X ):
    return np.real( np.fft.ifft( X ).flatten() )
    
# recover wav
def recover_wav( x, gt_x ):
    # pad zero to pred spectrogram if length is smaller than ground truth
    if len(x) < len(gt_x):
        n_freq = x.shape[1]
        pad = np.zeros( ( len(gt_x)-len(x), n_freq ) )
        x = np.concatenate( (x, pad), axis=0 )
    
    x = real_to_complex( x, gt_x )
    x = half_to_whole( x )
    s = ifft_to_wav( x )
    return s
    
# recover ground truth wav
def recover_gt_wav( x ):
    x = half_to_whole( x )
    s = ifft_to_wav( x )
    return s
    
    
### Write out wav
def write_wav( x, fs, path ):
    scaled = np.int16( x/np.max(np.abs(x)) * 16384. )
    wavio.write( path, scaled, fs, sampwidth=2)
    
    
### Main function
if __name__ == '__main__':
    if not os.path.exists( cfg.fe_fft_fd ): os.makedirs( cfg.fe_fft_fd )
    if not os.path.exists( cfg.results_fd ): os.makedirs( cfg.results_fd )
    if not os.path.exists( cfg.fe_fft_fd + '/mix' ): os.makedirs( cfg.fe_fft_fd + '/mix' )
    if not os.path.exists( cfg.fe_fft_fd + '/left' ): os.makedirs( cfg.fe_fft_fd + '/left' )
    if not os.path.exists( cfg.fe_fft_fd + '/right' ): os.makedirs( cfg.fe_fft_fd + '/right' )
    CalculateAllSpectrogram( cfg.fe_fft_fd )