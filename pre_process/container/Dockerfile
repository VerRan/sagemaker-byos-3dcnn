FROM python:3.7-slim-buster

RUN pip3 install keras tensorflow opencv-python-headless tqdm numpy
RUN mkdir dataset
# Add a Python script and configure Docker to run it
#  videoto3d.py classes.txt
ADD processing_script.py  videoto3d.py classes.txt / 
ENTRYPOINT ["python3", "/processing_script.py"]