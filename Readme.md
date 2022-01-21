# Thesis Progress Tracker

## Work in Progress

#### SFU Streaming
- [ ] Share TransformationTracks between multiple consumers
- [ ] Include tag in audio/video stream


#### Monitoring
- [x] Show absolute processing time in the SFU
- [ ] Determine RTT for producer &#8594; SFU &#8594; consumer


#### Triggers Lambdas
- [x] Run multiple ONNX models for image analysis
- [x] Define common interface for triggers
  - [ ] Face Recognition
  - [ ] Gender Detection
  - [ ] Age Detection
  - [ ] Car Plate Detection


#### Transformations Lambdas
- [ ] Define common interface for transformations
- [ ] Implement transformations with interface
  - [ ] blur_simple
  - [ ] blur_pixelate
  - [ ] replace_bar


#### Privacy model support
- [ ] Define model structure
- [ ] Compile and parse model
  - [ ] Compile structure
  - [ ] Compile transformation names
  - [ ] Compile trigger names
- [ ] Execute chain of transformations defined model


### Only for Thesis
- [ ] present formula for max time that a sfu can take to ensure stream in same fps. Interesting for later!</span>


#### Not important anymore?
    
- Ensure that videos from a producer are replayed in correct fps.... well stick to camera then because live streams are the most interesting part
  - however, for evaluation it could be interesting
- change webcam output format? Why could this be necessary?
  - for testing the maximum framerate my system can handle!
    - but I could also enqueue every frame twice/thrice/... and see if it can cope with this
    - How to add new models asynchronously?
