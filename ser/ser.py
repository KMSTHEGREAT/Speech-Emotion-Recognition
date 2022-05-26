from keras.models import model_from_json
import pandas as pd
import numpy as np
import sys
import warnings
import librosa

# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# loading json and model architecture 
json_file = open('saved_models/model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("saved_models/Ser_Model.h5")
#print("Loaded model from disk")

# the optimiser
loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

def Predict(name):
    sampling_rate=44100
    audio_duration=2.5
    input_length = sampling_rate * audio_duration
    n_mfcc = 30
    n =n_mfcc
    path = name

    data, _ = librosa.load(path, sr=sampling_rate
                                ,res_type="kaiser_fast"
                                ,duration=2.5
                                ,offset=0.5)

    X = np.empty(shape=(data.shape[0], n, 216, 1))

    # Random offset / Padding
    if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
        else:
            offset = 0
            data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
            # MFCC extraction 
            MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
            MFCC = np.expand_dims(MFCC, axis=-1)
            X = MFCC

    newdf = np.expand_dims(X, axis=0)
    newpred = loaded_model.predict(newdf, 
                            batch_size=16, 
                            verbose=1)

    lb = pd.read_csv('saved_models/labels.csv')
    final = newpred.argmax(axis=1)
    final = final.astype(int).flatten()

    return(lb['labels'][final])
