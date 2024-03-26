import numpy as np 
import time
import tensorflow as tf 
from tensorflow.python.ops import gen_audio_ops as audio_ops
import time
import pyaudio

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