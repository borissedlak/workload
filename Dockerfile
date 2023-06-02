# set base image (host OS)
FROM python:3.9-slim-buster

#Local dependencies for cv2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set the working directory in the container
WORKDIR /src/

# copy the content of the local directory to the working directory
COPY requirements.txt /src/
COPY sfu_server /src/
COPY model /src/
COPY detector /src/
COPY video_data /video_data/
COPY data /data/

# install dependencies
RUN pip install -r requirements.txt

#VOLUME /src/data

# command to run on container start
CMD [ "python", "./main.py" ]