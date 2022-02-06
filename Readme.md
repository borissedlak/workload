# Thesis Progress Tracker

## Work in Progress

### Evaluation 

- [x] Dockerize Server
- [x] Deploy Server in AWS
- [x] Measure RTT for Producer
- [x] Calculate run times for functions
- [x] Export run times as csv
- [ ] Measure run time for audio as well...

#### Nvidia Jetson

- [x] Setup Jetson environment
- [x] Clone project in Jetson
- [x] Setup ONNX environment
- [x] Run SFU there

#### SFU Streaming

- [x] Share TransformationTracks between multiple consumers
- [x] Include tag in audio/video stream

#### Monitoring

- [x] Show absolute processing time in the SFU
- [x] Show processing time of distinct operations
- [x] Determine RTT for producer &#8594; SFU &#8594; consumer

#### Triggers Lambdas

- [x] Run multiple ONNX models for image analysis
- [x] Define common interface for triggers
    - [x] Face Recognition
    - [x] Gender Detection
    - [x] Age Detection
    - [x] Car Plate Detection

#### Transformations Lambdas

- [x] Define common interface for transformations
- [x] Implement transformations with interface
    - [x] blur_pixelate
    - [x] replace_bar
    - [x] max_resize

#### Privacy model support

- [x] Define model structure
- [x] Compile and parse model
    - [x] Compile structure
    - [x] Compile transformation names
    - [x] Compile trigger names
- [x] Execute chain of transformations defined model
- [x] Update model during runtime keeping streams alive
- [x] Allow multiple chains in model
- [x] Select a fitting model with media_source and tag

### Only for Thesis

- [ ] Present formula for max time that a sfu can take to ensure stream in same fps. Interesting for later!
- [ ] After implementing all triggers and transformations, provide stats on their performance, also related to params
- [ ] Describe how and when images are encoded, have a little look at the codecs that are supported by WebRTC
- [ ] Compare run times for functions with and without GPU acceleration
- [ ] Do I want to measure run time for audio as well? I mean I implemented it; I want to write about it anyway for reasons of streaming
- [ ] Document transformation and trigger parameters in a nice presentable fashion
- [ ] Compare running time for different parameters in pixelate, ironically less anonymization effect

#### Not important anymore?

- Ensure that videos from a producer are replayed in correct fps.... well stick to camera then because live streams are
  the most interesting part
    - however, for evaluation it could be interesting
- change webcam output format? Why could this be necessary?
    - for testing the maximum framerate my system can handle!
        - but I could also enqueue every frame twice/thrice/... and see if it can cope with this
        - How to add new models asynchronously?
