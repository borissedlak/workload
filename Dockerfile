# set base image (host OS)
FROM python:3.10-slim-buster
ENV HTTP_SERVER=""
ENV DEVICE_NAME=""
ENV CLEAN_RESTART=""

#Local dependencies for cv2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install gcc python3-dev graphviz -y

# set the working directory in the container
WORKDIR /src/

# copy the content of t he local directory to the working directory
COPY requirements.txt /src/
COPY FGCS /src/
COPY model /src/
COPY detector /src/
COPY video_data /video_data/
COPY data /data/

# install dependencies TODO: Check if no-deps makes problems
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "python", "./agent.py" ]