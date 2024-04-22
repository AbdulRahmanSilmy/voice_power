import numpy as np 
import tensorflow as tf 
from tensorflow.io import gfile
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
import os
from typing import List

def _create_melspec(audio_signal: np.ndarray) -> np.ndarray:
    """
    Generates spectrogram base on the audio signal 

    Parameters
    ----------
    audio_signal: np.ndarray
        An array of shape (N,1) where N denotes the number of samples.

    Returns
    -------
    spectrograms: np.ndarray 
        An array of shape (H,W) where H and W denote the height and
        width of the spectrogram respectively.   
    """
    
    # normalise the audio_signal
    audio_signal = audio_signal - np.mean(audio_signal)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    # create the spectrogram
    spectrogram = audio_ops.audio_spectrogram(audio_signal,
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

def list_subfolders(folder_path: str) -> List[str]:
    """
    List the subfolders within a parent directory

    Parameters
    ----------
    folder_path: str
        The parent directory containing the folder paths 

    Returns
    -------
    subfolders: list[str]
    """
    subfolders = [f.path[len(folder_path)+1:] for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

# get all the files in a directory
def get_files(word: str, data_folder: str) -> List[str]:
    """
    get all the audio files paths in a directory

    Parameters
    ----------
    word: str
        The word from the speech dataset
    
    data_folder: str
        The parent folder containing the speech dataset

    Returns
    -------
    word_filenames: List[str]
        The list of audio filenames of a word.
    
    """
    word_filenames=gfile.glob(data_folder + '/'+word+'/*.wav')
    return word_filenames

def get_voice_position(audio: np.ndarray, noise_floor: float) -> np.ndarray:
    """
    Get location of voice 

    Parameters
    ----------
    audio: array_like
        An array of shape (N,) where N representst the number of samples.
        The audio data containing a voice command.

    noise_floor: float
        The percentage of the max value of audio to be considered a 
        signal 

    Return
    -------
    audio_edges: array_like
        An array of shape (2,) where it contains the start and end position 
        of the voice signal respectively. 
    """
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    audio_edges=tfio.audio.trim(audio, axis=0, epsilon=noise_floor)
    audio_edges=audio_edges.numpy()

    return audio_edges