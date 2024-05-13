# voice_power
Developing an AI voice activated power adapter

## Project Goals 

- The project aims to develop a robust wake-word detection system tailored to make home applications smart.
- The solution is to be deployed on an edge device (Raspberry Pi).
- The wake word model is developed to control the switching of a household appliance based on a spoken trigger word. 

## What is a wake word detection?

Wake word models, also referred to as trigger word models, play a pivotal role in voice-activated
systems, such as virtual assistants like Amazon Alexa or Google Assistant. These models are designed to
continuously listen to audio streams and detect specific words or phrases. Once the wake word is
recognized, the system activates to process subsequent user commands or queries.

![alt text](image-1.png)

## Overview 
Initially, the data pipeline continuously monitors an audio stream, capturing and storing the most recent one-second segment.
Subsequently, these audio segments undergo transformation into spectrograms, a fundamental feature
utilized by wake word detection models. These wake word models analyze the spectrograms to predict
the presence of the target word 'Marvin', with predictions exceeding 90% triggering a five-second
recording. This recording is then processed by a speech-to-text model hosted by a cloud service called
wit-ai, which not only transcribes the speech but also infers the corresponding action to be executed.
Finally, the inferred action is utilized to control or activate an appliance.
![alt text](image-2.png)

## Future work 
- Employ depthwise separable convolution to reduce in the wake word detection model.
- Dynamically adjust the length of the audio recording sent to the speech-to-text model based on the duration of user speech.  
- Deploy a light weight speech-to-text model locally on the edge.  

## Links 
- [Report](https://drive.google.com/file/d/11lDMEO4o5V73IKrxzBxsniu0r8q__0Lz/view?usp=sharing)
- [Video Presentation](https://www.youtube.com/watch?v=JU44qwh8rWc)
- [GitHub](https://github.com/AbdulRahmanSilmy/voice_power)