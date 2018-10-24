import os
import re
import wave
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import cPickle
#import python_speech_features as features
from scipy.spatial import distance
import scipy.io.wavfile as wav
import datetime
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from IPython.display import clear_output
from tqdm import *





def load_file(file_path):
    with open(file_path, 'rb') as cPickle_file:
        f = cPickle.load(cPickle_file)
        return f
        
def save_file(dataList, file_path):
    with open(file_path + '.pkl', 'wb') as cPickle_file:
        cPickle.dump(
                dataList, 
                cPickle_file, 
                protocol=cPickle.HIGHEST_PROTOCOL)

def load_dataset(file_path):
    data_set = load_file(file_path)
    X_train, y_train, info_train, X_test, y_test, info_test = data_set
    return X_train, y_train, info_train, X_test, y_test, info_test
  

phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
    "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
    "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]
    # 61 different phonemes

def get_total_duration(file):
    """Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
    for line in reversed(list(open(file))):
        [_, val, _] = line.split()
        return int(val)

def find_phoneme (phoneme_idx):
    for i in range(len(phonemes)):
        if phoneme_idx == phonemes[i]:
            return i
    print("PHONEME NOT FOUND, NaN CREATED!")
    print("\t" + phoneme_idx + " wasn't found!")
    return -1

def create_mfcc(method, filename):
    """Perform standard preprocessing, as described by Alex Graves (2012)
    http://www.cs.toronto.edu/~graves/preprint.pdf
    Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
    [1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)

    method is a dummy input!!"""


    (rate,sample) = wav.read(filename)
    print(sample[0])

    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
    preemph=0.97, appendEnergy=True)

    derivative = np.zeros(mfcc.shape)
    for i in range(1, mfcc.shape[0]-1):
        derivative[i, :] = mfcc[i+1, :] - mfcc[i-1, :]

    out = np.concatenate((mfcc, derivative), axis=1)

    return out, out.shape[0]

def calc_norm_param(X, VERBOSE=False):
    """Assumes X to be a list of arrays (of differing sizes)"""
    total_len = 0
    mean_val = np.zeros(X[0].shape[1])
    std_val = np.zeros(X[0].shape[1])
    for obs in X:
        obs_len = obs.shape[0]
        mean_val += np.mean(obs,axis=0)*obs_len
        std_val += np.std(obs, axis=0)*obs_len
        total_len += obs_len
    
    mean_val /= total_len
    std_val /= total_len

    if VERBOSE:
        print(total_len)
        print(mean_val.shape)
        print('  {}'.format(mean_val))
        print(std_val.shape)
        print('  {}'.format(std_val))

    return mean_val, std_val, total_len

def normalize(X, mean_val, std_val):
    for i in range(len(X)):
        X[i] = (X[i] - mean_val)/std_val
    return X

def set_type(X, type):
    for i in range(len(X)):
        X[i] = X[i].astype(type)
    return X


def preprocess_dataset(source_path, ignore_SA = True):
    """Preprocess data, ignoring compressed files and files starting with 'SA'"""
    i = 0
    X = []
    Y = []
    idx = []
    num_plot = 4

    for dirName, subdirList, fileList in os.walk(source_path):
        for fname in fileList:
            if not fname.endswith('.PHN') or (ignore_SA and fname.startswith("SA")): 
                continue

            phn_fname = dirName + '/' + fname
            wav_fname = dirName + '/' + fname[0:-4] + '.WAV'
            a = dirName + '/' + fname[0:-4]

            total_duration = get_total_duration(phn_fname)
            fr = open(phn_fname)

            X_val, total_frames = create_mfcc('DUMMY', wav_fname)
            total_frames = int(total_frames)

            X.append(X_val)
            
            y_val = np.zeros(total_frames) - 1
            
            start_ind = 0
            for line in fr:
                [start_time, end_time, phoneme] = line.rstrip('\n').split()
                start_time = int(start_time)
                end_time = int(end_time)

                phoneme_num = find_phoneme(phoneme)
                end_ind = int(np.round((end_time)/total_duration*total_frames))
        
                y_val[start_ind:end_ind] = phoneme_num

                start_ind = end_ind
            fr.close()

            if -1 in y_val:
                print('WARNING: -1 detected in TARGET')
                print(y_val)

            Y.append(y_val.astype('int32'))

            i+=1
            print(i, end=' ', flush=True)

            b = re.search('[^/]*$',source_path)[0]
            a = a[len(source_path)-len(b):]
            print(a)
            data_set = re.search('^[^/]*',a)[0]
            a = a[len(data_set)+1:]
            dialect = re.search('^[^/]*',a)[0]
            a = a[len(dialect)+1:]
            gender = a[0]
            a = a[1:]
            speaker = re.search('^[^/]*',a)[0]
            a = a[len(speaker)+1:]
            sentence = a
#             print(data_set, dialect, gender, speaker, sentence)
            
            idx.append([data_set, dialect, gender, speaker, sentence])
    
    idx = np.array(idx)
    idx = pd.DataFrame(idx)
    idx.columns = ['data_set', 'dialect', 'gender', 'speaker', 'sentence']
    
    print()
    return X, Y, idx

def runPreprocess(pfile, tfile):
    data_type = 'float32'

    # paths                 = path_reader('path_toke.txt')
    train_source_path    = os.path.join(pfile, 'TRAIN')
    test_source_path    = os.path.join(pfile, 'TEST')
    target_path            = os.path.join(pfile, tfile)

    ##### PREPROCESSING #####
    print()

    print('Preprocessing data ...')
    print('This will take a while ...')
    X_train, y_train, info_train = preprocess_dataset(train_source_path, ignore_SA = False)
    X_test, y_test, info_test =preprocess_dataset(test_source_path, ignore_SA = True)

    print('Preprocessing done.')


    print()
    print('Normalizing data ...')
    print('    Each channel mean=0, sd=1 ...')

    mean_val, std_val, _ = calc_norm_param(X_train+X_test)

    X_train = normalize(X_train, mean_val, std_val)
    X_test = normalize(X_test, mean_val, std_val)

    X_train = set_type(X_train, data_type)
    X_test = set_type(X_test, data_type)

    print('Saving data ...')
    print('   ', target_path)
    with open(target_path + '.pkl', 'wb') as cPickle_file:
        cPickle.dump(
            [X_train, y_train, info_train, X_test, y_test, info_test], 
            cPickle_file, 
            protocol=cPickle.HIGHEST_PROTOCOL)

    print('All done!')
    print()

def pltDistanceDistribution(a, distFunction = distance.euclidean , bins = 50, flatten = True,  **kargs):

    if(not flatten):
        start = datetime.datetime.now()
        distFunction(a[0], a[1])
        end = datetime.datetime.now()
        t = end - start
    else:
        start = datetime.datetime.now()
        distFunction(a[0].flatten(), a[1].flatten())
        end = datetime.datetime.now()
        t = end - start
    num_iters = int(100 / t.total_seconds())
    
    x = []

    if(flatten):
        for i in range(num_iters):
            x.append(distFunction(a[random.randint(0,len(a)-1)].flatten(), a[random.randint(0,len(a)-1)].flatten()))
    else:
        for i in range(num_iters):
            x.append(distFunction(a[random.randint(0,len(a)-1)], a[random.randint(0,len(a)-1)]))
    plt.hist(x, bins = 50)
    plt.show()

def dec2onehot(dec, output_dim = 61):
    ret=[]
    for u in dec:
        assert np.all(u<output_dim)
        num=u.shape[0]
        r=np.zeros((num,output_dim))
        r[range(0,num),u]=1
        ret.append(r)
    return np.array(ret)

def ctx(data, ctx_fr = 5):
    ret=[]
    for utt in data:
        l=utt.shape[0]
        ur=[]
        for t in range(l):
            f=[]
            for s in range(t-ctx_fr,t+ctx_fr+1):
                if(s<0) or (s>=l):
                    f.append(np.zeros(utt[0].shape))
                else:
                    f.append(utt[s,:])
            ur.append(f)
        ret.append(np.array(ur))
    return np.array(ret)

def getMaxLenth(a):
    max_len = 0
    for i in a:
        if len(i) > max_len:
            max_len = len(i)
    return max_len

def load_trainingStats(fname, col = 'val_acc'):
    with open(fname, 'rb') as cPickle_file:
        a = cPickle.load(cPickle_file)
    return a[col]

def compareTrainingStats(accList, labelList):
    plt.figure(figsize=(22,12))
    assert len(accList) == len(labelList)
    for i in range(len(accList)):
        plt.plot(accList[i], label = labelList[i])
    plt.legend()
    plt.show()

def accByCategories(model, xtest = None, ytest = None , info_test = None, preProcessedFile = None):
    if(preProcessedFile):
        print('Loading preprocessed test data from file ...')
        with open(preProcessedFile, 'rb') as cPickle_file:
            Xgroups, ygroups = cPickle.load(cPickle_file)

    else:
        print('Computing test data ...')
        info_test.columns = ['set', 'region', 'gender', 'sperker', 'sentence']
        X_t = np.array(xtest)
        y_t = np.array(ytest)

        Xgroups = []
        ygroups = []

        for i in ["DR1", "DR2", "DR3","DR4", "DR5", "DR6","DR7", "DR8"]:
            for j in ["M", "F"]:
                query = 'region =="' + i +'" and gender == "' + j + '"'
                a = info_test.query(query).index
                b = np.concatenate(ctx(X_t[list(a)], 5))
                c = np.concatenate(dec2onehot(y_t[list(a)]))
                Xgroups.append(b)
                ygroups.append(c)

        print('Test data computing done.')
        print('Saving ...')

        with open('./testDataByCategories' + '.pkl', 'wb') as cPickle_file:
            cPickle.dump(
                [Xgroups, ygroups], 
                cPickle_file, 
                protocol=cPickle.HIGHEST_PROTOCOL)
        
        print('Done saving.')

    acc = []
    for i in range(len(Xgroups)):        
        _, a = model.test_on_batch(Xgroups[i], ygroups[i])
        acc.append(a)
    
    return acc
        
def accByPhonemes(model, x = None, y = None, preProcessedFile = None):
    if(preProcessedFile):
        print('Loading preprocessed test data from file ...')
        with open(preProcessedFile, 'rb') as cPickle_file:
            phonemeGroups, xt, yy = cPickle.load(cPickle_file)

    else:
        print('Computing test data ...')
        xt = np.concatenate(ctx(x, 5))
        yy = np.concatenate(dec2onehot(y))
        yt = np.concatenate(y)
        phonemeGroups = []
        for i in range(0, 61):
            phonemeGroups.append(list(np.argwhere(yt == i).flatten()))

        print('Test data computing done.')
        print('Saving ...')

        with open('./testDataByPhonemes' + '.pkl', 'wb') as cPickle_file:
            cPickle.dump(
                [phonemeGroups, xt, yy], 
                cPickle_file, 
                protocol=cPickle.HIGHEST_PROTOCOL)

        print('Done saving.')
    

    acc = []
    for i in phonemeGroups:
        _, a = model.test_on_batch(xt[i], yy[i])
        acc.append(a)
        
    return acc

def plotByCategory(a, b, nameList):

    sets = ('1M', '1F', '2M', '2F', '3M', '3F', '4M', '4F', '5M', '5F', '6M', '6F', '7M', '7F', '8M', '8F')

    plt.rcParams['figure.figsize'] = (22, 12)
    bar_width = 0.35
    index = np.arange(len(a))
    rects1 = plt.bar(index, a, bar_width, color='#0072BC', label=nameList[0])
    rects2 = plt.bar(index + bar_width, b, bar_width, color='#ED1C24', label=nameList[1])
    plt.xticks(index + bar_width, sets, rotation = 45)
    plt.legend()
    plt.show()

def plotByPhonemes(a, b, nameList):

    plt.rcParams['figure.figsize'] = (22, 12)
    bar_width = 0.35
    index = np.arange(len(phonemes))
    rects1 = plt.bar(index, a, bar_width, color='#0072BC', label=nameList[0])
    rects2 = plt.bar(index + bar_width, b, bar_width, color='#ED1C24', label=nameList[1])
    plt.xticks(index + bar_width, phonemes, rotation = 55)
    plt.legend()
    plt.show()

def resize(p, r=5):
    l = p.shape[0]
    w = p.shape[1]
    ret = np.zeros([l*r, w*r, 3])
    for i in range(l):
        for j in range(w):
            ret[i*r:r*(i+1), j*r:r*(j+1), :] = p[i,j]
            
    return ret
    
def getFeats(X_train):
    model = VGG16(include_top=False, input_shape=(55,65,3))
    feats = []
    num = 0
    tr = np.arange(len(X_train))
    for i in tqdm(tr):
        a = resize(X_train[i][:,:13], 5)
        b = model.predict(a[np.newaxis,:,:,:])
        feats.append(b[0,0,:,:])
#        clear_output()
        
    feats = np.array(feats)
    return feats





