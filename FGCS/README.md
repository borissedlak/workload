To Alireza:

1. only operate in /FGCS folder, the rest is irrelevant
2. except for the requirements.txt, which you must install
3. run FogNode.py class in background of the same host, this ensures that the metrics are reported correctly
4. supply env parameters to agent.py, namely DEVICE_NAME = Laptop, and optionally CLEAN_RESTART = True
5. run agent.py, if CLEAN_RESTART = TRUE, it will start from scratch, if not, it will use the existing model for $DEVICE_NAME

In the background it will start to process frames and log the surprise according to the currently trained model to Fog/slo_stream_results.csv
If you want you can export the current model by pressing 'e' in the agent window and use this model instead of starting from scratch
One way to change the distributions is by pressing '+' or '-', because internally it increases the number of threads processed
But if you want to change the processing duration, you might change the fps or pixel parameter directly, e.g., by overwriting the config in agent.py:110