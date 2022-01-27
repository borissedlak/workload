# set base image (host OS)
FROM python:3.9-slim-buster

# set the working directory in the container
WORKDIR /src/

# copy the content of the local directory to the working directory
COPY requirements.txt /src/
COPY sfu_server /src/
COPY model /src/
COPY detector /src/

# install dependencies
RUN pip install -r requirements.txt


# command to run on container start
CMD [ "python", "./server.py" ]