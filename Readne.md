## Next Steps
- Implement the remote sdp functions so that the client 
  can then connect to the sfu and stream a video
  - add controller to client
  - in a second stage it should be able to send webcam streams as well
    and not only video files. If it shows that it does not make any difference
    regarding the performance I can always stick to recorded sample video
  - I would need these videos anyway to make a comparison between the different setups!
- Ensure that videos from a producer are replayed in correct fps
- Transform audio as well? How to detect something here?
### Monitoring
  - show fps of consumed video
  - show processing time in the sfu
    - include that in a rtt
    - <span style="color:red">how to actually act if fps is dropped? Create backpressure(?) or start to drop frames?
      - I would rather not start to implement things here since this is not related to how the model is applied but to stream related features that would cost me a lot of time
    - <span style="color:orange">present formula for max time that a sfu can take to ensure stream in same fps</span>
### Coffe
  - <span style="color:red">how to build models? Could I do it myself? It would be cool but would take some time, but it would be an awesome example
  - <span style="color:red">is there maybe another model publicly available?
  - <span style="color:red">how to add new models asynchronously?
### Lambdas
  - <span style="color:red">how to actually call lambda functions? 
    - Maybe indiced array of functions?