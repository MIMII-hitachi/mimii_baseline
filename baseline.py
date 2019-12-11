#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo TANABE (Center for Technology Innovation - Digital Technology, Hitachi Ltd.)
 Coptright (C) 2019 Center for Technology Innovation - Digital Technology, Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019.
"""

# load parameter.yaml
########################################################################
import yaml
with open("baseline.yaml") as stream: param = yaml.load(stream)
"""
The parameters are loaded as a dict-type.
# default value
base_directory : ./dataset
pickle_directory: ./pickle
model_directory: ./model
result_directory: ./result
result_file: result.yaml

audio:
  bit : 16
  sr: 16000
  sec : 10.
  channels : 8

feature:
  bin: 64
  frame : 5

librosa:
  n_fft: 1024
  hop_length: 512
  n_mels: 64
  power: 2.0

fit:
  compile:
    optimizer : adam
    loss : mean_squared_error
  epochs : 50
  batch_size : 512
  shuffle : True
  validation_split : 0.1
  verbose : 1
########################################################################
"""

# setup STD I/O
########################################################################
import logging
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
"""
STD msg will be output as the "baseline.log" file.
########################################################################
"""

# import default python-library
########################################################################
import pickle
import os
import sys
import glob
########################################################################

# import additional python-library
########################################################################
import numpy
numpy.random.seed(0) # set seed
import keras
import librosa
import librosa.core
import librosa.feature
# from import
from tqdm import tqdm
from sklearn import metrics
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense
########################################################################

## versions ##
__versions__ = "1.0.0"
##############

########################################################################
# visualizer
class visualizer(object):
    def __init__(self):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt= plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)
    def loss_plot(self,loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1,1,1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Epoch")
        ax.legend(["Train", "Test"], loc="upper right")
    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        self.plt.savefig(name)
########################################################################

########################################################################
# file I/O
########################################################################
# mkdir
def try_mkdir(dirname, silence=False):
    """
    generate output directory.

    dirname : str
        directory name.
    silence : bool
        boolean setting for STD I/O.
    return : None
    """
    try:
        os.mkdir(dirname)
        if not silence:
            print("%s dir is generated"%dirname)
    except:
        if not silence:
            print("%s dir is exist"%dirname)
        else:
            pass

# pickle I/O
def save_pickle(filename, data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info( "save_pickle -> {}".format(filename) )
    with open(filename, 'wb') as f:
        pickle.dump(data , f)
def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info( "load_pickle <- {}".format(filename) )
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# wav file Input
def file_load(wav_name, sampling_rate=16000, mono=False):
    """
    load a .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for monoral data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=sampling_rate, mono=mono)
    except:
        logger.error( f'{"file_broken or not exists!! : {}".format(wav_name)}' )
def demux_wav(wav_name, sampling_rate=16000, channel=0 ):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed monoral data
    """
    try:
        multi_channel_data, sr = file_load(wav_name, sampling_rate=sampling_rate)
        return numpy.array(multi_channel_data)[channel,:]
    except ValueError as f:
        logger.warning(f'{f}')
########################################################################

########################################################################
# feature extractor
########################################################################
def file_to_vectorarray(    filename,
                            sampling_rate=16000,
                            n_mels=64, 
                            frame=5, 
                            n_fft=1024,
                            hop_length=512,
                            power=2.0,
                            **kwargs):
    """
    convert filename to vector array.

    filename : str
        target .wav file
    sampling_rate : int ( default = 16000 )
        sampling rate [hz] of target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 culc. the number of dimensions
    dims = n_mels * frame
    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    mel_spectrogram_temp = librosa.feature.melspectrogram(  y = demux_wav( filename, sampling_rate=sampling_rate ), 
                                                            n_fft = n_fft, 
                                                            hop_length=hop_length, 
                                                            n_mels=n_mels, 
                                                            power=power,
                                                            **kwargs)
    # 03 convert melspectrogram to log mel energy
    mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram_temp + sys.float_info.epsilon)
    # 04 culc. total vector size
    vectorarray_size = len(mel_spectrogram[0,:]) - frame + 1
    # 05-1 Skip too short clip
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)
    # 05-2 Multiframes
    vectorarray = numpy.empty((vectorarray_size, 0), float)
    # 06 gegnerate vectors
    for t in range(frame):
        vectorarray = numpy.concatenate((vectorarray, mel_spectrogram[:, t : t + vectorarray_size].T), axis = 1)
    return vectorarray
def list_to_dataset(file_list, msg = "culc...",
                    sampling_rate=16000,
                    n_mels=64,
                    frame=5, 
                    power=2.0, 
                    **kwargs):
    """
    convert the file_list to dataset.
    in this function, file_to_vectorarray is looped, and concatenated.

    file_list : list [ str ]
        .wav filename list of dataset

    msg : str ( default = "culc..." )
        description for tqdm.
        this parameter will be input into "desc" param @ tqdm.
    sampling_rate : int ( default = 16000 )
        sampling rate [hz] of target .wav file

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, fearture_vector_length)
    """
    # 01 culc. the number of dimensions
    dims = n_mels * frame
    # 02 initialize the dataset
    dataset = numpy.empty( (0, dims ), float )
    # 03 loop of file_to_vectorarray
    for filename in tqdm( file_list, desc=msg ):
        vectorarray = file_to_vectorarray( filename, 
                                            sampling_rate=sampling_rate,
                                            n_mels=n_mels,
                                            frame=frame, 
                                            power=power, 
                                            **kwargs )
        dataset = numpy.concatenate((dataset, vectorarray ), axis=0)
    return dataset
def dataset_generator(  target_dir, train_pickle, eval_pickle, eval_meta_pickle,
                        normal="normal", abnormal="abnormal", ext="wav",
                        sampling_rate=16000,
                        n_mels=64,
                        frame=5, 
                        power=2.0, 
                        **kwargs):
    """
    target_dir : str
        base directory path of the dataset
    train_pickle : str
        training dataset .pickle file name 
    eval_pickle : str
        evaluation filename list .pickle file name 
    eval_meta_pickle : str
        evaluation meta info list .pickle file name 

    normal : str (default="normal")
        directory name the normal data located
    abnormal : str (default="abnormal")
        directory name the abnormal data located
    ext : str (default="wav")
        filename extension of the target audio file 
    sampling_rate : int ( default = 16000 )
        sampling rate [hz] of target .wav file

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, fearture_vector_length)
        eval_dataset_list : list [ str ]
            evaluation filename list
        eval_meta_list : list [ boolean ] 
            evaluation meta info. list
            * normal/abnnormal = 0/1,
    """
    logger.info( "target_dir : {}".format(target_dir) )

    # 01 normal list generate
    normal_list = sorted(glob.glob( "{dir}/{normal}/*.{ext}".format( dir=target_dir, normal=normal, ext=ext ) ))
    normal_meta = numpy.zeros(len(normal_list))
    if len(normal_list) == 0: logger.exception(f'{"no_wav_data!!"}')

    # 02 abnormal list generate
    abnormal_list = sorted(glob.glob( "{dir}/{abnormal}/*.{ext}".format( dir=target_dir, abnormal=abnormal, ext=ext ) ))
    abnormal_meta = numpy.ones(len(abnormal_list))
    if len(abnormal_list) == 0: logger.exception(f'{"no_wav_data!!"}')

    # 03 separate train & eval
    train_dataset_list = normal_list[len(abnormal_list):]
    train_meta_list = normal_meta[len(abnormal_list):]
    eval_dataset_list  = numpy.concatenate((normal_list[:len(abnormal_list)], abnormal_list), axis=0)
    eval_meta_list = numpy.concatenate((normal_meta[:len(abnormal_list)], abnormal_meta), axis=0)
    logger.info( "train_file num : {num}".format( num=len(train_dataset_list) ) )
    logger.info( "eval_file  num : {num}".format( num=len(eval_dataset_list)  ) )

    # 04 generate data
    train_data = list_to_dataset(   train_dataset_list,
                                    msg = "generate train_dataset",
                                    sampling_rate=sampling_rate,
                                    n_mels=n_mels,
                                    frame=frame, 
                                    power=power, 
                                    **kwargs )
    ID = numpy.random.permutation(len(train_data[:,0]))
    train_data = train_data[ID,:]

    # 05 save pickle
    save_pickle(train_pickle, train_data)
    save_pickle(eval_pickle, eval_dataset_list)
    save_pickle(eval_meta_pickle, eval_meta_list)

    return train_data, eval_dataset_list, eval_meta_list
########################################################################

########################################################################
# keras model
########################################################################
def keras_model(inputDim):
    """
    declare the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64).
    """
    inputLayer = Input(shape = (inputDim, ))
    h = Dense(64, activation = "relu")(inputLayer)
    h = Dense(64, activation = "relu")(h)
    h = Dense(8, activation = "relu")(h)
    h = Dense(64, activation = "relu")(h)
    h = Dense(64, activation = "relu")(h)
    h = Dense(inputDim, activation=None)(h)
    return Model(inputs = inputLayer, outputs = h)
########################################################################

if __name__ == "__main__":

    # generate output directory
    try_mkdir( param["pickle_directory"] )
    try_mkdir( param["model_directory"] )
    try_mkdir( param["result_directory"] )

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = sorted(glob.glob( "{base}/*/*/*".format( base=param["base_directory"] ) ))

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for num, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=num+1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine}_{id}_{db}.pickle".format(pickle=param["pickle_directory"], machine=machine_type, id=machine_id, db=db)
        eval_pickle  = "{pickle}/eval_file_list_{machine}_{id}_{db}.pickle".format(pickle=param["pickle_directory"], machine=machine_type, id=machine_id, db=db)
        eval_meta_pickle  = "{pickle}/eval_meta_list_{machine}_{id}_{db}.pickle".format(pickle=param["pickle_directory"], machine=machine_type, id=machine_id, db=db)
        model_name = "{model}/model_{machine}_{id}_{db}.hdf5".format(model=param["model_directory"], machine=machine_type, id=machine_id, db=db)
        history_png = "{model}/history_{machine}_{id}_{db}.png".format(model=param["model_directory"], machine=machine_type, id=machine_id, db=db)
        evaluation_result_key = "{machine}_{id}_{db}".format(machine=machine_type, id=machine_id, db=db)

        # dataset generator
        print("==== DATASET_GENERATOR ====")
        if os.path.exists(train_pickle) and os.path.exists(eval_pickle) and os.path.exists(eval_meta_pickle):
            train_data = load_pickle(train_pickle)
            eval_dataset_list = load_pickle(eval_pickle)
            eval_meta_list = load_pickle(eval_meta_pickle)
        else:
            train_data, eval_dataset_list, eval_meta_list = dataset_generator( target_dir, train_pickle, eval_pickle, eval_meta_pickle,
                                                                                sampling_rate=param["audio"]["sr"],
                                                                                n_mels=param["feature"]["bin"],
                                                                                frame=param["feature"]["frame"], 
                                                                                **param["librosa"] )
        # model training
        print("==== MODEL TRAINING ====")
        model = keras_model( param["feature"]["bin"] * param["feature"]["frame"] )
        model.summary()
        # training
        if os.path.exists(model_name): model.load_weights(model_name)
        else:
            model.compile(**param["fit"]["compile"])
            history = model.fit(    train_data, 
                                    train_data,
                                    epochs = param["fit"]["epochs"],
                                    batch_size = param["fit"]["batch_size"],
                                    shuffle = param["fit"]["shuffle"],
                                    validation_split = param["fit"]["validation_split"],
                                    verbose = param["fit"]["verbose"])
            visualizer.loss_plot( history.history["loss"], history.history["val_loss"] )
            visualizer.save_figure(history_png)
            model.save_weights(model_name)

        # evaluation
        print("==== EVALUATION ====")
        y_pred = [ 0. for k in eval_meta_list ]
        y_true = eval_meta_list
        for num, filename in tqdm( enumerate(eval_dataset_list), total=len(eval_dataset_list) ):
            try:
                data = file_to_vectorarray( filename,
                                            sampling_rate=param["audio"]["sr"],
                                            n_mels = param["feature"]["bin"],
                                            frame = param["feature"]["frame"],
                                            **param["librosa"])
                pred_temp = numpy.mean(numpy.square( data - model.predict( data ) ), axis=1)
                y_pred[num] =  numpy.mean(pred_temp)
            except:
                logger.warning( "File broken !! : {}".format(filename) )
        score = metrics.roc_auc_score(y_true, y_pred)
        logger.info( "AUC : {}".format(score) )
        evaluation_result["AUC"] = float(score)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    print("\n===========================")
    logger.info( "all results -> {}".format(result_file) )
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
