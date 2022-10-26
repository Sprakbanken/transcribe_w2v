# A CLI for running wav2vec2 on larger audio files
This repo contains `transcription_pipeline`, a CLI for transcribing larger audiofiles with wav2vec2. By default, it transcribes the audio file to Norwegian Bokmål, but other models can be specified. It also runs speaker diarization on the file. `.csv` is the default output format, but it can also produce `.eaf` or `srt`. 

## Installation
```
pip install -r requirements.txt
python -m build
pip install dist/w2vtranscriber*.whl
```

## Usage
To transcribe an audiofile in Bokmål: `python -m w2vtranscriber.transcription_pipeline transcribe path/to/audiofile path/to/output/csv`

To get more detailed documentation, run: `python -m w2vtranscriber.transcription_pipeline --help`

This script is in very early stages of development and may not always work as intended.
