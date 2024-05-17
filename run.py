"""
This script runs the wake word detection module. It loads the model, sets 
the threshold for detection, and runs the wake word detection module.

Note
----
This script requires the entering of the Kasa account details and the Wit client
"""
import asyncio
import numpy as np
import keras
from modules.wake_word import wake_word_detection 

#enter kasa account details here 
USERNAME="XXX"
PASSWORD="XXX"
IP_ADDRESS="XXX"

#enter wit client info here 
WIT_CLIENT="XXX"

#load model
MODEL_PATH="models/model_training5.keras"
model=keras.models.load_model(MODEL_PATH)

#set threshold for detection
THRESHOLD=0.90


if __name__=="__main__":
    asyncio.run(wake_word_detection(model,
                                    USERNAME,
                                    PASSWORD,
                                    IP_ADDRESS,
                                    WIT_CLIENT,
                                    threshold=THRESHOLD,
                                    num_chunks=np.inf))