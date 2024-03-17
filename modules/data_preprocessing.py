from tensorflow.keras.utils import Sequence
import numpy as np 
from tensorflow.io import gfile
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
import tensorflow as tf

class DataGenerator(Sequence):
    """Generates data for Keras
        Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, filenames,classes,data_folder,target='marvin',
                 to_fit=True, batch_size=32,
                 shuffle=True,training=True,noise_floor=0.1):
     
        self.filenames = filenames
        self.classes=classes
        self.data_folder=data_folder
        self.target=target
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training=training
        self.noise_floor=noise_floor

        self.on_epoch_end()

    def __len__(self):
     
        return int(np.floor(len(self.down_sampled_filenames) / self.batch_size))

    def __getitem__(self, index):
    
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        filenames_temp = [self.down_sampled_filenames[k] for k in indexes]

        # Generate data
        X = self._generate_X(filenames_temp)

        if self.to_fit:
            y = self._generate_y(filenames_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.training:
            self.down_sampled_filenames=self._down_sample_data()
        else:
            self.down_sampled_filenames=np.concatenate(self.filenames)
        self.indexes = np.arange(len(self.down_sampled_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _down_sample_data(self):
        target_class_filenames=self.filenames[0]
        down_sampled_filenames=[target_class_filenames]
        num_samples_per_remaining_class=int(len(target_class_filenames)/(len(self.classes)-1))
        for class_filenames in self.filenames[1:]:
            class_down_sample = np.random.choice(class_filenames,size=num_samples_per_remaining_class,replace=False)
            down_sampled_filenames.append(class_down_sample)

        down_sampled_filenames=np.concatenate(down_sampled_filenames)
        return down_sampled_filenames

    def _generate_X(self, filenames_temp):
    
        # Initialization
        X = []

        # Generate data
        for i, filename in enumerate(filenames_temp):
            # Store sample
            X.append(self._generate_melspec(filename))

        X=np.array(X)
        return X

    def _generate_y(self, filenames_temp):
   
        y = np.empty((self.batch_size,), dtype=int)

        # Generate data
        for i, filename in enumerate(filenames_temp):
            # Store sample
            if self.target in filename:
                y[i,] = 1
            else:
                y[i,] = 0
                
        return y
    


        # process a file into its spectrogram
    def _process_audio(self,file_path):
        # load the audio file
        audio_tensor = tfio.audio.AudioIOTensor(file_path)
        # convert the audio to an array of floats and scale it to betweem -1 and 1
        audio = tf.cast(audio_tensor[:], tf.float32)
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        # randomly reposition the audio in the sample
        voice_start, voice_end = get_voice_position(audio, self.noise_floor)
        end_gap=len(audio) - voice_end
        random_offset = np.random.uniform(0, voice_start+end_gap)
        audio = np.roll(audio,-random_offset+end_gap)

        # add some random background noise
        background_volume = np.random.uniform(0, 0.1)
        # get the background noise files
        background_files = get_files('_background_noise_',self.data_folder)
        background_file = np.random.choice(background_files)
        background_tensor = tfio.audio.AudioIOTensor(background_file)
        background_start = np.random.randint(0, len(background_tensor) - 16000)
        # normalise the background noise
        background = tf.cast(background_tensor[background_start:background_start+16000], tf.float32)
        background = background - np.mean(background)
        background = background / np.max(np.abs(background))
        # mix the audio with the scaled background
        audio = audio + background_volume * background

        # get the spectrogram
        return audio
    
    def _generate_melspec(self,filename):
        audio=self._process_audio(filename)
        # normalise the audio
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        # create the spectrogram
        spectrogram = audio_ops.audio_spectrogram(audio,
                                                  window_size=320,
                                                  stride=160,
                                                  magnitude_squared=True).numpy()
        # reduce the number of frequency bins in our spectrogram to a more sensible level
        spectrogram = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, 6],
            strides=[1, 6],
            pooling_type='AVG',
            padding='SAME')
        spectrogram = tf.squeeze(spectrogram, axis=0)
        spectrogram = np.log10(spectrogram + 1e-6)
        return spectrogram
        


def split_dataset(filenames,target,classes,train_ratio,val_ratio,test_ratio,shuffle=True):
    if (train_ratio+val_ratio+test_ratio)!=1:
        raise ValueError('All ratios must sum to one')
    
    marvin_bool = [True if target in filename else False for filename in filenames]
    filename_index=np.arange(len(filenames))
    marvin_index = filename_index[marvin_bool]
    
    if shuffle:
        np.random.shuffle(marvin_index)
    train_num=int(np.floor(len(marvin_index)*train_ratio))
    val_num=int(np.floor(len(marvin_index)*val_ratio))
    test_num=int(np.floor(len(marvin_index)*test_ratio))
   
    train_filenames=[filenames[marvin_index][:train_num]]
    val_filenames=[filenames[marvin_index][train_num:train_num+val_num]]
    test_filenames=[filenames[marvin_index][train_num+val_num:]]

    for word in classes:
        if word is target:
            continue
        class_bool = [True if word in filename else False for filename in filenames]
        class_index = filename_index[class_bool]
        if shuffle:
            np.random.shuffle(class_index)
        num_remaining_class=len(classes)-1
        num_test_sample_class=int(test_num/num_remaining_class)
        num_val_sample_class=int(val_num/num_remaining_class)

        class_test,class_val,class_train=np.split(class_index,[num_test_sample_class,
                                                               num_test_sample_class+num_val_sample_class])
        train_filenames.append(filenames[class_train])
        val_filenames.append(filenames[class_val])
        test_filenames.append(filenames[class_test])

    return train_filenames,val_filenames,test_filenames


# get all the files in a directory
def get_files(word,data_folder):
    return gfile.glob(data_folder + '/'+word+'/*.wav')

# get the location of the voice
def get_voice_position(audio, noise_floor):
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    return tfio.audio.trim(audio, axis=0, epsilon=noise_floor)

