# voice_power
Developing an AI voice activated power adapter

## Background and Motivations

The project aims to create a practical implementation rather than just existing on a
notebook, focusing on daily usability. The goal is to develop an AI voice-activated
switch, that allows users to control appliances with a spoken trigger word. This will
initially be applied to a bedroom lamp.

## Objectives

Utilize signal processing and deep learning to handle the continuous and
unstructured audio data. Signal processing will reduce background noise and create
mel spectrograms, while deep learning models will classify the wake word. Key
optimization metrics include model accuracy and detection latency for real-time
performance.

## Planned Activities 

- Conduct a literature review on signal processing and deep learning models for wake word detection on resource-constrained edge devices.
- Source datasets for training the deep learning model.
- Deploy the deep learning model on a Raspberry Pi.
- Implement electric circuitry with relays to create the switch.
- Test the real-time performance of the system and iterate for improvements.
- If time permits, attempt deployment on a microcontroller like the ESP32 by scaling down the model using quantization techniques, considering its lower cost for commercial viability.