from cProfile import run
import pandas as pd
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyannote.audio import Pipeline

from inaSpeechSegmenter import Segmenter
import pympi
from pathlib import Path
from datetime import timedelta
from srt import Subtitle, compose
import argparse
import torch

# Utils


def predict(filepath, trancriptionfunc):
    """Transcribe an audio file at a filepath given a
    transcription function

    Parameter
    ----------
    filepath
        the path to the audio file to be transcribed
    transcriprtionfunc
        a transcription function which returns a string
        given an audio file input

    returns: a string with the predicted transcription"""

    return trancriptionfunc(filepath)


def transcribe_series(filename_series, transcriptionfunc, path_to_audio):
    """Transcribe each file in a pandas Series with filenames, given a transcription function.

     Parameter
    ----------
    filename_series
        a Pandas Series with filenames, possibly filepaths, if the files are grouped in directories
    transcriptionfunc
        a transcription function which returns a string
        given an audio file input
    path _to_audio
        path to the folder where the audio files are in the series are located

    returns: a series with predicted transcriptions"""
    return filename_series.apply(
        lambda x: predict(path_to_audio + x, transcriptionfunc)
    )


# Transcribe with wav2wec
def wav2vec_transcribe(
    filepath, processor, model, offset, duration, device, limit=30, print_output=False
):
    """Transcribe an audiofile or segment of an audio file with wav2vec.

    Parameter
    ----------
    filepath
        path to the audio file
    processor
        a wav2vec processor, e.g. Wav2Vec2ProcessorWithLM.from_pretrained('NbAiLab/nb-wav2vec2-1b-bokmaal')
    model
        a wav2vec model, e.g. Wav2Vec2ForCTC.from_pretrained('NbAiLab/nb-wav2vec2-1b-bokmaal')
    offset
        where to start transcribing, in seconds from start of file
    duration
        the duration of the audio segment, in seconds from the offset, which should be transcribed.
    device
        the device the process should be run on (cpu of gpu)
    limit=30:
        The max amount of seconds accepted for a segment
    print_output= False
        Option to print the transcriptions to terminal

    return: the predicted transcription of the audio segment
    """

    try:
        if duration > limit:
            if print_output:
                print("")
            return ""
        else:
            audio, rate = librosa.load(
                filepath, sr=16000, offset=offset, duration=duration
            )
            input_values = processor(
                audio, sampling_rate=rate, return_tensors="pt"
            ).input_values.to(device)
            logits = model(input_values).logits.cpu()
            transcription = processor.batch_decode(logits.detach().numpy()).text
            if print_output:
                print(transcription[0])
            return transcription[0]
    except Exception as e:
        print(e)
        if print_output:
            print("_")
        return "_"


def transcribe_df_w2v(
    df, processor, model, device, audio_dir=None, print_output=False, outfile=None
):
    """Transcribe audio with wav2vec given a DataFrame with segments. A column 'wav2vec' will
    be created with the predicted transcriptions.

    Parameter
    ----------
    df
        a DataFrame with segments
    processor
        a wav2vec processor, e.g. Wav2Vec2ProcessorWithLM.from_pretrained('NbAiLab/nb-wav2vec2-1b-bokmaal')
    model
        a wav2vec model, e.g. Wav2Vec2ForCTC.from_pretrained('NbAiLab/nb-wav2vec2-1b-bokmaal')
    device
        the device the process should be run on (cpu or gpu)
    audio_dir=None
        a directory where the files in the 'audio_path' column in the df are located
        if this column does not contain complete paths
    print_output= False
        Option to print the transcriptions to terminal
    outfile=None
        the path to an csv file that the transcribed DataFrame is stored to

    return: a DataFrame with transcriptions in the column 'wav2vec'if outfile is None,
    else create a csv file with the name specified in outfile"""
    if audio_dir is None:
        df.loc[:, "wav2vec"] = df.apply(
            lambda row: wav2vec_transcribe(
                row.audio_path,
                processor,
                model,
                row.start,
                row.duration,
                device,
                print_output=print_output,
            ),
            axis=1,
        )
    else:
        df.loc[:, "wav2vec"] = df.apply(
            lambda row: wav2vec_transcribe(
                audio_dir + row.audio_path,
                processor,
                model,
                row.start,
                row.duration,
                device,
                print_output=print_output,
            ),
            axis=1,
        )
    if outfile is None:
        return df
    else:
        df.to_csv(outfile, index=False)


# Create a segmentation df with speaker annotation
def diarize(audiofile, outfile=None):
    """Identify all the individual speakers in an audio file and return
    a DataFrame with segments with start and end codes and speaker tags.

    Parameter
    ----------
    audiofile
        the adiofile to diarize
    outfile=None
        the path to an csv file that the diarized DataFrame is stored to

    Return: a DataFrame with columns 'speaker', 'start', 'end', 'duration',
    and 'audio_path' if outfile is None, else create a csv file with the
    name specified in outfile
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    # parameters = pipeline.default_parameters() # might be possible to get shorter segments by adjusting params
    # parameters["min_duration_off"] = 0.001
    # parameters["onset"] = 0.9
    # pipeline.instantiate(parameters)

    diarization = pipeline(audiofile)
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({"speaker": speaker, "start": turn.start, "end": turn.end})
    df = pd.DataFrame(result)
    df.loc[:, "duration"] = df.end - df.start
    df.loc[:, "audio_path"] = audiofile
    if outfile is None:
        return df
    else:
        df.to_csv(outfile, index=False)


def run_vad(
    audiofile,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    window_size_samples=1536,
    speech_pad_ms=30,
    return_seconds=True,
    outfile=None,
):
    """Run voice activity detection on an audiofile.

    Parameter
    ----------
    audiofile
        the adiofile to run VAD on
    outfile=None
        the path to an csv file that the diarized DataFrame is stored to

    Return: a DataFrame with columns 'speaker', 'start', 'end', 'duration',
    and 'audio_path' if outfile is None, else create a csv file with the
    name specified in outfile
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    model
    get_ts, read_audio = utils[0], utils[2]
    audio_tns = read_audio(audiofile)
    vad = get_ts(
        audio_tns,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        speech_pad_ms=speech_pad_ms,
        return_seconds=return_seconds,
    )

    df = pd.DataFrame(vad)
    df.loc[:, "duration"] = df.end - df.start
    df.loc[:, "audio_path"] = audiofile
    if outfile is None:
        return df
    else:
        df.to_csv(outfile, index=False)

    # Identify background
    def identify_background(audiofile, complete=False, outfile=None):
        """Identify noise, music, male and female speakers in an audio file.
        The identified segments may be quite large, so this script is not useful
        for pre-asr segmentation, but can be used to extract metadata about the
        audio file.

        Parameter
        ----------
        audiofile
            the audiofile to be analyzed
        complete=False
            if False, include info about music and noise only,
            not speech
        outfile=None
            the path to an csv file that the background DataFrame is stored to


        Return: a DataFrame with columns 'type', 'start', 'end', 'duration',
        and 'audio_path' if outfile is None, else create a csv file with the
        name specified in outfile
        """
        seg = Segmenter()
        segmentation = seg(audiofile)
        segdicts = [{"type": x[0], "start": x[1], "end": x[2]} for x in segmentation]
        df = pd.DataFrame(segdicts)
        df.loc[:, "duration"] = df.end - df.start
        df.loc[:, "audio_path"] = audiofile
        if not complete:
            df = df[df.type.isin(["music", "noise"])]
        if outfile is None:
            return df
        else:
            df.to_csv(outfile, index=False)


# Output to eaf with or without background tier
def _ann_to_tier(tier, start, end, annotation, eafobj):
    if end - start > 0:
        eafobj.add_annotation(
            tier, int(start * 1000), int(end * 1000), value=annotation
        )


def transcription_df_to_eaf(trans_df, audiopath, outfile, background_df=None):
    """Convert a DataFrame with transcriptions to an eaf-file, possibly with a DataFrame with info about
    background sounds (music, noise). A tier in the eaf file is created for each speaker, as well as a background
    tier, if relevant. A segmented, but not diarized DataFrame is passed, a column 'transcription' is converted
    to a similarly named tier.

    Parameter
    ----------
    trans_df
        a DataFrame with transcriptions
    audiopath
        the path to the transcribed audiofile
    outfile
        the name of the eaf output file
    background_df=None
        if assigned to a DataFrame created by `identify_backgrounds`, a background tier is created in the
        eaf file

    return: an eaf file with one tier per speaker and possibly a background tier. This eaf file has a pointer
    to the audiofile
    """
    trans_df.loc[:, "wav2vec"] = trans_df.wav2vec.fillna("_")
    eafob = pympi.Elan.Eaf()
    audiopath_abs = str(Path(audiopath).absolute())
    audiopath_rel = str(Path(audiopath).relative_to("./"))
    eafob.add_linked_file(audiopath_abs, relpath=audiopath_rel)
    if "speaker" in trans_df.columns:
        for speaker in trans_df.speaker.unique():
            speaker_df = trans_df[trans_df.speaker == speaker]
            eafob.add_tier(speaker)
            speaker_df.apply(
                lambda row: _ann_to_tier(
                    speaker, row.start, row.end, row.wav2vec, eafob
                ),
                axis=1,
            )
    else:
        eafob.add_tier("transcription")
        trans_df.apply(
            lambda row: _ann_to_tier(
                "transcription", row.start, row.end, row.wav2vec, eafob
            ),
            axis=1,
        )
    if background_df is not None:
        eafob.add_tier("background")
        background_df.apply(
            lambda row: _ann_to_tier("background", row.start, row.end, row.type, eafob),
            axis=1,
        )
    eafob.to_file(outfile)


# Output to subtext file


def _row_to_srt(row):
    start = timedelta(seconds=row.start)
    end = timedelta(seconds=row.end)
    content = row.wav2vec
    index = row.index
    return Subtitle(index=index, start=start, end=end, content=content)


def transcription_to_subtext(trans_df):
    """Convert a transcription DataFrame to a subtext file. This function
    is for now rather basic, without speaker names, and there may be overlapping
    segments, which will not look good as subtitles"""
    return compose(list(trans_df.apply(lambda row: _row_to_srt(row), axis=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for transcribing larger audiofiles with wav2vec2."
    )
    parser.add_argument(
        "mode",
        help="Choose the mode: 'vad', 'diarize', 'transcribe', 'background', 'convert'",
    )
    parser.add_argument("audiofile", help="Specify the path to an audiofile")
    parser.add_argument("outfile", help="Specify the path to an output file")
    parser.add_argument(
        "-v",
        "--verbose",
        help="print transcribed sentences to terminal",
        action="store_true",
    )
    #    parser.add_argument(
    #        "-b",
    #        "--background",
    #        action="store_true",
    #        help="Identify segments with music and noise. For now, the background is only added to eaf files",
    #    )
    parser.add_argument(
        "-d",
        "--diarize",
        help=(
            "Run speaker diarization in addition to voice activity detection in transcribe mode. "
            "Be aware that this option may lead to segments > 30 seconds, which will be filtered out "
            "by the ASR"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--format",
        nargs="?",
        default="csv",
        help=(
            "Specify the file format of a transcription file."
            "Possible options are csv, eaf or srt. Defaults to csv."
            "The eaf and srt options are only implemented for the 'transcribe' and 'convert' modes."
            "If no format is specified in 'convert' mode, an eaf file is produced"
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        default="NbAiLab/nb-wav2vec2-1b-bokmaal",
        help=(
            "Specify which model to use. "
            "By default, 'NbAiLab/nb-wav2vec2-1b-bokmaal' is used."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="?",
        default=None,
        help=(
            "Specify a presegmented input csv instead of running diarization. "
            "The csv must have a column 'audio_path' with the path to the audio files. "
            "If the paths are not absolute or relative to the current directory, use '--audio_dir'."
            "Use this option also when converting a csv to eaf or srt."
        ),
    )
    parser.add_argument(
        "-a",
        "--audio_dir",
        nargs="?",
        default=None,
        help=(
            "If an input file is specified with --input, and the 'audio_path' "
            "column is not an absolute path or path relative to the current path, provide the path to the folder where "
            "the audio files in 'audio_path' are located"
        ),
    )
    args = parser.parse_args()

    if args.mode == "diarize":
        print(f"Diarizing {args.audiofile} to {args.outfile}")
        diarize(args.audiofile, outfile=args.outfile)
    elif args.mode == "vad":
        print(f"Running voice activity detection on {args.audiofile} to {args.outfile}")
        run_vad(args.audiofile, outfile=args.outfile)
    #    elif args.mode == "background":
    #        print(f"Identifying background in {args.audiofile} to {args.outfile}")
    #        identify_background(args.audiofile, outfile=args.outfile)
    elif args.mode == "transcribe":
        #        background = None
        #        if args.background:
        #            print(f"Identifying noise and music in {args.audiofile}")
        #            background = identify_background(args.audiofile)
        vad_df = None
        if args.input is None:
            if args.diarize:
                print(f"Diarizing {args.audiofile}...")
                vad_df = diarize(args.audiofile)
            else:
                print(f"Running VAD on {args.audiofile}...")
                vad_df = run_vad(args.audiofile)
        else:
            vad_df = pd.read_csv(args.input)
        print_output = False
        if args.verbose:
            print_output = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model)
        model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
        print(f"Transcribing to {args.outfile}")
        trans_df = transcribe_df_w2v(
            vad_df,
            processor,
            model,
            device,
            audio_dir=args.audio_dir,
            print_output=print_output,
        )
        if args.format == "eaf":
            transcription_df_to_eaf(
                trans_df, args.audiofile, args.outfile, background_df=background
            )
        elif args.format == "srt":
            with Path(args.outfile).open(mode="w") as f:
                f.write(transcription_to_subtext(trans_df))
        else:
            trans_df.to_csv(args.outfile, index=False)
    elif args.mode == "convert":
        trans_df = pd.read_csv(args.input)
        trans_df = trans_df.dropna(subset="wav2vec")
        print(f"converting {args.input} to {args.outfile}")
        if args.format == "srt":
            with Path(args.outfile).open(mode="w") as f:
                f.write(transcription_to_subtext(trans_df))
        else:
            transcription_df_to_eaf(
                trans_df, args.audiofile, args.outfile
            )  # TODO: Add background option
