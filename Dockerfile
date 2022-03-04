FROM huggingface/transformers-tensorflow-gpu



# copy all the files to the container
COPY ./transcription_pipeline.py .

ARG username

# set a directory for the app
WORKDIR /home/$username

RUN useradd -u 8877 $username

# Change to non-root privilege
USER $username

# Build: sudo docker build --build-arg username=$USER -t test01 .
# Run: sudo docker run --rm test01 id