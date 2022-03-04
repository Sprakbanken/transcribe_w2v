# A CLI for running wav2vec2 on larger audio files
This repo contains `transcription_pipeline`, a CLI for transcribing larger audiofiles with wav2vec2. By default, it transcribes the audio file to Norwegian Bokmål, but other models can be specified. It also runs speaker diarization on the file. `.csv` is the default output format, but it can also produce `.eaf` or `srt`

To transcribe an audiofile in Bokmål: `python -m transcription_pipeline transcribe path/to/audiofile path/to/output/csv`

To get more detailed documentation, run: `python -m transcription_pipeline --help`

This script is in very early stages of development and may not always work as intended.

There is also a Dockerfile in the repo for building a Docker image for the transcription pipeline. This has not been tested yet. The Dockerfile is made to run as the current user, not as root.

To build the image: `sudo docker build --build-arg username=$USER -t test01  .`
To run: `sudo docker run -it test01`