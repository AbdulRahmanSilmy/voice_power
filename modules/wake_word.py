import numpy as np 
import time
import tensorflow as tf 
from tensorflow.python.ops import gen_audio_ops as audio_ops
import time
import pyaudio
from kasa import Discover, Credentials
import wave
from .utils import _create_melspec
from wit import Wit




class AudioBuffer():
    """
    A buffer that stores chunks of audio in a sequential manner. 

    Parameters
    ----------
    model: keras.model
        A pretrained keras model 
    
    thresh: float, default=0.5
        The cut-off probability to denote a positive class prediction
    
    chunk_dur: float, default=0.5
        The time in seconds denoting size of the chunk.
    
    sample_rate: int, default=16000
        The sample rate used for recording audio.
    
    buffer_size: int, default=5
        The number of chunks stores in the buffer. 
    
    Attributes
    -----------
    chunk_size: int
        The number of samples in a chunk. Determined by chunk_dur*sample_rate
    
    buffer: np.ndarray 
        An array of shape (N,) where N denotes the number of samples stored in 
        a buffer. N=buffer_size*chunk_size

    Notes
    -----
    In the future introduce a finite state machine at the process_audio_stream 
    method to make the detection of wake word more stable. 
    """
 
    def __init__(self, model,thresh=0.5,chunk_dur=0.5, sample_rate=16000, buffer_size=5):
        self.model=model
        self.thresh=thresh
        self.chunk_size = int(chunk_dur*sample_rate)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size * self.chunk_size))


    def _update_buffer(self, chunk: np.ndarray):
        """
        A new chunk of audio is loaded into a buffer in a first in 
        last out manner. 

        Parameters
        ----------
        chunk: np.ndarray
            A chunk of audio of shape (chunk_size,) that gets inserted into 
            the buffer.
        """
        
        num_samples=chunk.shape[0]
        self.buffer[:-num_samples] = self.buffer[num_samples:]
        self.buffer[-num_samples:] = chunk

    def _detection(self,audio_signal: np.ndarray) -> bool: 
        """
        Checking if wake word is present in the audio signal

        Parameters
        -----------
        audio_signal: np.ndarray
             An audio signal of shape (N,) where N denotes the number of 
             samples.
        
        Returns
        -------
        presense: bool
            True if presense of wake work is detected False otherwise. 
        """
       
        mel_spec=_create_melspec(audio_signal.reshape(-1,1))
        prediction=self.model(mel_spec.reshape(1,99,43,1),training=False)
        prediction=prediction.numpy()[0][0]
        if prediction>self.thresh:
            presense= True 
        else:
            presense=False

        return presense 

    def process_audio_stream(self, chunk:np.ndarray) -> bool:
        """
        Updates the buffer and checks if older chunks in buffer 
        contain the wakeword. 

        Parameters
        ----------
        chunk: np.ndarray
            A chunk of audio of shape (chunk_size,) that gets inserted into 
            the buffer.
        
        detection: bool
            True if presense of wake work is detected False otherwise.
        """
        
        self._update_buffer(chunk)
        detection=self._detection(self.buffer[-3*self.chunk_size:-self.chunk_size])
        
        return detection

def record_audio(output_file, duration=5, chunk_size=1024, format=pyaudio.paInt16, channels=2, sample_rate=44100):
    audio = pyaudio.PyAudio()

    

    # Open audio stream
    stream = audio.open(format=format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)

    #print("Recording...")

    frames = []

    # Record audio for the specified duration
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    #print("Finished recording.")

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Write the recorded audio to a WAV file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

async def wake_word_detection(model,
                              audio_path="test_audio/file1.wav",
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        chunk_dur=0.5,
                        buffer_size=5,
                        threshold=0.5,
                        num_chunks=20,
                        delay=3):
    """
    A temporary function to test the AudioBuffers wake word detection
    """
   
    start=time.time()

    device = await Discover.discover_single(
        "192.168.147.35",
        credentials=Credentials("abdul.rahman.silmy@gmail.com", "Lenovo4lif@"),
        discovery_timeout=10
    )
    await device.turn_off()

    #initializing client 
    client = Wit("PWTMDL7K22YQYG7UYMDN5Q4IIGEKZTAX")

    CHUNK = int(rate*chunk_dur)
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=CHUNK)


    buffer=AudioBuffer(model,thresh=threshold,chunk_dur=chunk_dur,sample_rate=rate,buffer_size=buffer_size)

    print('Recording in...')

    delay_val=0

    state=False
    past_detection=False

    while delay_val<num_chunks:
        data=stream.read(CHUNK, exception_on_overflow = False)
        
        npdata=np.frombuffer(data,dtype=np.int16)
  
        #introducing delay to tackle mic noise at the start of recordings
        if delay_val<delay:
            print(delay-delay_val)
        else:
            detection=buffer.process_audio_stream(npdata) 

            if detection and not past_detection:
                record_audio(audio_path)
                resp = None
                with open(audio_path, 'rb') as f:
                  resp = client.speech(f, {'Content-Type': 'audio/wav'})
                value=resp['traits']['wit$on_off'][0]['value']
                message=resp['text']
                print(f'message: {message} | action: {value}')
                
                if value=='on':
                    await device.turn_on()
                else:
                    await device.turn_off()
            
            #print(detection)
            past_detection=detection

        delay_val+=1

    await device.turn_off()
    print('Done')
    stream.close()
    p.terminate()

    end=time.time()
    print(end-start)



        


