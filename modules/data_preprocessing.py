"""
Contains the classes and functions usesd for data preprocessing 

The module structure is the following:

- The ``DataGenerator`` is the class used to generate batches of data to be used for 
  training with tensorflow models. This is akin to ImageDataGenerator from keras but 
  instead converts audio wav files to 2d array constituting the melspecs. 
  It solves class imbalance where the minority class is the target class 
  by performing dynamic downsampling of the non-target class. It also applies random 
  augmentations the audio file before it gets converted to mel spectrograms. 

- The ``SplitDataset`` is the class used to split the speech commands datasets from 
  multiclass to a binary class dataset while solving class imbalance for testing and
  training datasets. 

- The ``FilterAudioData`` filters through all the audio files for a word in the 
  dataset to find valid files.

- The ``process_background`` takes in background noise and segments it to one 
  second audio signals to be added to the non-target class for wake word detection.

"""

# import from public libraries
import os
from typing import Optional, Tuple, List
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf
import scipy
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# imports from modules
from .utils import get_voice_position, get_files, _create_melspec


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras Sequence based data generator. 
    Suitable for building data generator for training and prediction.
    """

    def __init__(self, filenames, classes, data_folder, target='marvin',
                 to_fit=True, batch_size=32,
                 shuffle=True, training=True, noise_floor=0.1,target_ratio=0.5,background_ratio=0.1):

        self.filenames = filenames
        self.classes = classes
        self.data_folder = data_folder
        self.target = target
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.noise_floor = noise_floor
        self.target_ratio = target_ratio
        self.background_ratio = background_ratio
        self.other_ratio=1-self.target_ratio-self.background_ratio
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(len(self.down_sampled_filenames) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

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
            self.down_sampled_filenames = self._down_sample_data()
        else:
            self.down_sampled_filenames = np.concatenate(self.filenames)
        self.indexes = np.arange(len(self.down_sampled_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _down_sample_data(self):
        target_class_filenames = self.filenames[0]
        down_sampled_filenames = [target_class_filenames]

        remaining_class_ratios=[self.background_ratio,self.other_ratio]

        num_samples_per_remaining_class=[int(ratio*len(target_class_filenames)) for ratio in remaining_class_ratios]

        for class_filenames,num_samples in zip(self.filenames[1:],num_samples_per_remaining_class):
            class_down_sample = np.random.choice(
                class_filenames, size=num_samples, replace=False)
            down_sampled_filenames.append(class_down_sample)

        down_sampled_filenames = np.concatenate(down_sampled_filenames)
        return down_sampled_filenames

    def _generate_X(self, filenames_temp):
        # Initialization
        X = []

        # Generate data
        for i, filename in enumerate(filenames_temp):
            # Store sample
            X.append(self._generate_melspec(filename))

        X = np.array(X)
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

    def _process_audio(self, file_path):
        # load the audio file
        audio_tensor = tfio.audio.AudioIOTensor(file_path)
        # convert the audio to an array of floats and scale it to betweem -1 and 1
        audio = tf.cast(audio_tensor[:], tf.float32).numpy()
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        # randomly reposition the audio in the sample
        voice_start, voice_end = get_voice_position(audio, self.noise_floor)
        voice_start=voice_start[0]
        voice_end=voice_end[0]
        end_gap = len(audio) - voice_end
        random_offset = int(np.random.uniform(0, voice_start+end_gap))
        audio = np.roll(audio, -random_offset+end_gap)

        # add some random background noise
        background_volume = np.random.uniform(0, 0.1)
        # get the background noise files
        background_files = get_files('_background_noise_', self.data_folder)
        background_file = np.random.choice(background_files)
        background_tensor = tfio.audio.AudioIOTensor(background_file)
        background_start = np.random.randint(0, len(background_tensor) - 16000)
        # normalise the background noise
        background = tf.cast(
            background_tensor[background_start:background_start+16000], tf.float32)
        background = background - np.mean(background)
        background = background / np.max(np.abs(background))
        # mix the audio with the scaled background
        audio = audio + background_volume * background

        # get the spectrogram
        return audio

    def _generate_melspec(self, filename):
        audio = self._process_audio(filename)
        spectrogram = _create_melspec(audio)
        return spectrogram


class SplitDataset():
    """
    Taking a multiclass dataset and converting it to a binary class dataset that is 
    split into train, test and validation sets. This class allows you to control 
    the ratio of dataset compromising the target class. Moreover the ratio of the 
    dataset consituting the background sound of the non-target class can also be controlled.

    Parameters
    ----------
    dict_filenames: dict
        A dictionary where every key corresponds to the different classes in the
        multiclass dataset and the values correspond the the filepaths to the 
        audio file associated to that class.

    target_class: str
        The name of the class that would be the target. 

    background_class: str
        The name of the class that would be the background. 

    target_ratio: float 
        The ratio of the split datasets that correspond to the target class. 

    background_ratio: float
        The ratio of the split datasets that correspond to the background samples 
        as part of the non-target class.

    train_ratio: float,
        The ratio of the whole dataset making up the training set. 

    val_ratio: float,
        The ratio of the whole dataset making up the validation set.

    test_ratio: float,
        The ratio of the whole dataset making up the testing set. 

    shuffle: bool, default=True
        If true denotes that the dataset will be shuffled before the split.

    seed: Optional[int], default=None 
        If not None, it sets the seed so that you can reproduce results if 
        the shuffle parameter is true. 

    Attributes
    ----------
    train_size: int
        The number of samples in the training dataset. 

    val_size: int
        The number of samples in the validation dataset.

    test_size: int 
        The number of samples in the testing dataset.  

    """

    def __init__(self,
                 dict_filenames: dict,
                 target_class: str,
                 background_class: str,
                 target_ratio: float,
                 background_ratio: float,
                 train_ratio: float,
                 val_ratio: float,
                 test_ratio: float,
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        self.dict_filenames = dict_filenames
        self.target_class = target_class
        self.background_class = background_class
        self.target_ratio = target_ratio
        self.background_ratio = background_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.train_size=None
        self.val_size=None
        self.test_size=None

    def _compute_split_size(self) -> Tuple[int]:
        """
        Determines the size of each train, test and val split based on 
        the number of target class samples. This is because when a multi-class dataset
        is converted to a binary class dataset the target class will make up the minority class.

        Returns
        --------
        train_size: int
            The number of samples in the training dataset. 

        val_size: int
            The number of samples in the validation dataset.

        test_size: int 
            The number of samples in the testing dataset.  
        """

        target_filenames = self.dict_filenames[self.target_class]
        num_target_audio = len(target_filenames)

        train_size = int(self.train_ratio*num_target_audio/self.target_ratio)
        test_size = int(self.test_ratio*num_target_audio/self.target_ratio)
        val_size = int(self.val_ratio*num_target_audio/self.target_ratio)

        return train_size, val_size, test_size

    def _word_split_size(self,
                         words: List[str],
                         word_ratio: float,
                         train_size: int,
                         val_size: int,
                         test_size: int) -> Tuple[int]:
        """
        Determines the number of samples in each test, validation and train datasets
        such that the ratio of the words in the test and validatio split matches the 
        word_ratio.

        Parameters
        ----------
        words: List[str]
            Contains the string of the words that together would make up a certain ratio 
            of each train, test and validation sets. 

        word_ratio: float
            Controls the ratio of the samples that corresponds to words present in the 
            test and validatio sets. The remaining samples go to the training dataset and 
            does not follow the word ratio. 

        train_size: int
            The size of the training set

        val_size: int 
            The size of the validation dataset

        test_size: int 
            The size of the test dataset

        Returns
        -------
        word_train_size: int 
            The number of samples in the train dataset corresponding to words.

        word_val_size: int 
            The number of samples in the validation dataset corresponding to words.

        word_test_size: int 
            The number of samples in the test dataset corresponding to words.

        """

        total_size = sum([train_size, val_size, test_size])

        words_size = sum([len(self.dict_filenames[word]) for word in words])

        if words_size < total_size*word_ratio:
            raise ValueError("Not enough samples to support splitting")

        word_train_size = int(train_size*word_ratio)
        word_val_size = int(val_size*word_ratio)
        word_test_size = int(test_size*word_ratio)

        return word_train_size, word_val_size, word_test_size

    def _get_word_splits(self,
                         words: List[str],
                         ratio: float,
                         do_stratify: bool = False) -> Tuple[List[str]]:
        """
        Splits the filenames corresponding to the speech command in words from dict_filenames
        into the train, validation and test splits. 

        Parameters
        ----------
        words: List[str]
            The speech commands used you want to split 

        ratio: float
            The ratio of each train, validation and test split that make up all the speech commands
            in words.

        do_stratify: bool, default=False
            Perform statified sampling of each speech commands of from words. 

        Returns
        --------
        word_train: List[str]
            The filepaths of the audio files corresponding to words in the train set.     

        word_val: List[str]
            The filepaths of the audio files corresponding to words in the validation set.     

        word_test: List[str]
            The filepaths of the audio files corresponding to words in the test set.     
        """
        filenames = np.concatenate(
            [self.dict_filenames[word] for word in words])

        if do_stratify:
            stratify = [file.split(os.path.sep)[1] for file in filenames]
        else:
            stratify = None

        word_sizes = self._word_split_size(
            words, ratio, self.train_size, self.val_size, self.test_size)

        word_rem, word_test = train_test_split(
            filenames, test_size=word_sizes[2], stratify=stratify)
        if do_stratify:
            stratify = [file.split(os.path.sep)[1] for file in word_rem]
        word_train, word_val = train_test_split(
            word_rem, test_size=word_sizes[1], stratify=stratify)

        return word_train, word_val, word_test

    def split(self) -> dict:
        """
        Takes the dict_filenames passed during initiatiating of the class and splits 
        it into train, validation and test sets. Each split makes up the key and value 
        pair of the dictionary that is returned. 

        Returns
        -------
        dict_split: dict
            The dictionary contains the following keys:

            dict_split['train'] -> filepaths from the target, background and the 
            remaining speech commands making up the training set
            dict_split['val'] -> filepaths from the target, background and the 
            remaining speech commands making up the validation set
            dict_split['test'] -> filepaths from the target, background and the 
            remaining speech commands making up the testing set

        """

        self.train_size, self.val_size, self.test_size = self._compute_split_size()

        target_train, target_val, target_test = self._get_word_splits(
            [self.target_class], self.target_ratio)
        background_train, background_val, background_test = self._get_word_splits(
            [self.background_class], self.background_ratio)

        other_class_ratio = 1-self.target_ratio-self.background_ratio
        other_classes = [word for word in self.dict_filenames.keys() if (
            word is not self.background_class) and (word is not self.target_class)]

        other_train, other_val, other_test = self._get_word_splits(
            other_classes, other_class_ratio)

        dict_split = {}
        dict_split['train'] = [target_train, background_train, other_train]
        dict_split['val'] = [target_val, background_val, other_val]
        dict_split['test'] = [target_test, background_test, other_test]

        return dict_split


class FilterAudioData():
    """
    Filters through all the audio files for a word in the dataset to find valid files

    Parameters
    ----------
    data_folder: str
        The folder path containing audio data of speech commands dataset

    noise_floor: float, default=0.1
        The percentage above max noise considered to contain voice command

    required_length: int, default=4000
        The minimum samples for a voice command to be considered a valid 
        sample.

    expected_samples: int, default=16000
        The number of samples required of the entire audio file to be considered 
        a valid file. 
    """

    def __init__(self,
                 data_folder: str,
                 noise_floor: float = 0.1,
                 required_length: int = 4000,
                 expected_samples: int = 16000):
        self.data_folder = data_folder
        self.noise_floor = noise_floor
        self.required_length = required_length
        self.expected_samples = expected_samples

    def _is_voice_present(self, audio_data: np.ndarray) -> bool:
        """
        Determines if voice is present by checking if the number of samples 
        for a voice command is above the desired length 

        Parameters
        ----------
        audio_data: np.ndarray
            An array of shape (N,) where N denotes the number of samples in 
            the audio data.

        Returns
        -------
        voice_presence: bool
            True indicating that a voice is present

        """
        position = get_voice_position(audio_data, self.noise_floor)
        voice_length = position[1] - position[0]
        voice_presence = voice_length >= self.required_length

        return voice_presence

    def _is_file_valid(self, file_path: str) -> bool:
        """
        Checks if an audio file is valid by having the expected number of samples
        and seeing if a voice is present. 

        Parameters
        ----------
        file_path: str
            The file path to the audio file 

        Returns
        --------
        bool
            True if audio file is valid 
        """
        # load the audio file
        audio_tensor = tfio.audio.AudioIOTensor(file_path)
        # convert the audio to an array of floats and scale it to betweem -1 and 1
        audio = tf.cast(audio_tensor[:], tf.float32)
        # check file has exactly 1s of data
        audio_length = audio.numpy().shape[0]
        if audio_length != self.expected_samples:
            return False

        # is there any voice in the audio?
        if not self._is_voice_present(audio):
            return False

        return True

    def get_valid_audio(self, word: str) -> List[str]:
        """
        Finds the valid filenames for a specific word in the speech commands data set

        Parameters
        ----------
        word: str
            The word in the speech commands dataset

        Returns
        --------
        file_names: List[str]
            The valid filenames of a word in the dataset
        """

        file_names = []
        for file_name in tqdm(get_files(word, self.data_folder), desc="Checking", leave=False):
            if self._is_file_valid(file_name):
                file_names.append(file_name)

        return file_names


def process_background(background_file_path, save_folder_path, num_samples=16000):
    """
    Takes in background noise and segments it to one second audio signals 
    to be added to the non-target class for wake word detection.

    Parameters
    ----------
    background_file_path: str
        The filepath associated to the background noise to be segmented.

    save_folder_path: str
        The folder path to store segmented audio files. 

    num_samples: int, default=16000
        The number of samples within a segmented audio file. 

    """
    _, audio = scipy.io.wavfile.read(background_file_path)
    audio_length = len(audio)

    background_type = background_file_path.split(os.path.sep)[-1][:-4]

    for i, section_start in enumerate(range(0, audio_length-num_samples, num_samples)):
        section_end = section_start + num_samples
        section = audio[section_start:section_end]
        sample_filename = os.path.join(
            save_folder_path, background_type+str(i)+".wav")
        scipy.io.wavfile.write(sample_filename, 16000,
                               section.astype(np.int16))
