import gc

import numpy as np
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read
from tqdm import tqdm

import whisper
from intervaltree import IntervalTree
from collections import Counter

whisper.DecodingOptions(fp16=False)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def reset_cuda(model):
    del model
    gc.collect()

    torch.cuda.empty_cache()


def init_diarizer(
        diarizer_model="pyannote/speaker-diarization-3.1",
        token=None
    ):
    model = Pipeline.from_pretrained(diarizer_model, use_auth_token=token).to(torch.device(device))
    return model


def init_whisper(
        asr_model="large-v3",
        token=None
    ):
    model = whisper.load_model(asr_model).to(device)
    return model


def preprocess(inputs, sampling_rate):
    if isinstance(inputs, str):
        with open(inputs, "rb") as f:
            inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, sampling_rate)

    if isinstance(inputs, dict):
        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != sampling_rate:
            inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, sampling_rate).numpy()

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


class TQDMProgressHook:
    """Hook to show progress of each internal step

    Parameters
    ----------
    transient: bool, optional
        Clear the progress on exit. Defaults to False.

    Example
    -------
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    with ProgressHook() as hook:
       output = pipeline(file, hook=hook)
    """

    def __init__(self, transient: bool = False):
        self.transient = transient

    def __enter__(self):
        self.progress = dict()
        return self

    def __exit__(self, *args):
        for progress in self.progress.values():
            progress.close()

    def __call__(
        self,
        step_name,
        step_artifact,
        file = None,
        total = None,
        completed=None,
    ):
        if completed is None:
            completed = total = 1

        if not hasattr(self, "step_name") or step_name != self.step_name:
            self.progress[step_name] = tqdm(total=total, desc=step_name, leave=False)

        self.progress[step_name].update(completed)

        # force refresh when completed
        if completed >= total:
            self.progress[step_name].refresh()


def diarize(
        diarizer_pipeline,
        diarizer_inputs,
        sampling_rate="",
        ):

    with TQDMProgressHook() as hook:
        diarization = diarizer_pipeline({
            "waveform": diarizer_inputs.to(device),
            "sample_rate": sampling_rate
        }, hook=hook)

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'label': label
        })

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append({
                "start": prev_segment["start"],
                "end": cur_segment["start"],
                "speaker": prev_segment["label"],
            })
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append({
        "start": prev_segment["start"],
        "end": cur_segment["end"],
        "speaker": prev_segment["label"],
    })

    return new_segments


def transcribe(
    asr_pipeline,
    inputs,
    segments,
    language='ja',
    task='transcribe',
    count=False
):

    t = IntervalTree()

    timestamps = []
    for segment in segments:
        speaker, start, end = segment['speaker'], segment['start'], segment['end']
        
        timestamps.append(start)
        timestamps.append(end)
        t.addi(start, end, speaker)

    output = asr_pipeline.transcribe(
        inputs,
        verbose=False,
        clip_timestamps=timestamps,
        language=language,
        task=task
    )

    preds = []
    if count:
        speakers = Counter()

    text = output['text']

    for segment in output['segments']:
        start, end = segment['start'], segment['end']
        overlaps = t.envelop(start, end)
        if not overlaps:
            overlaps = t.at((start + end) // 2)

        if overlaps:
            interval = min(overlaps, key=lambda iv: iv.end-iv.begin)
            speaker = interval.data
        else:
            speaker = speakers.most_common(1)[0][0]

        preds.append({
            'speaker': speaker,
            'text': segment['text'],
            'timestamp': (segment['start'], segment['end'])
        })

        if count:
            speakers[speaker] += segment['end'] - segment['start']

    if count:
        return text, preds, speakers.most_common()

    return text, preds
