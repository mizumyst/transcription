import gc

import numpy as np
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from numba import cuda

def reset_cuda(model):
    del model
    gc.collect()

    device = cuda.get_current_device()
    device.reset()


def init_diarizer_pipeline(
        diarizer_model="pyannote/speaker-diarization-3.1",
        token=None
    ):
    diarizer_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=token)
    return diarizer_pipeline


def init_asr_pipeline(
        asr_model="openai/whisper-large-v3",
        token=None
):
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        chunk_length_s=30,
        token=token,
        batch_size=24,
        return_timestamps=True
    )
    return asr_pipeline


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


def diarize(
        diarizer_pipeline,
        diarizer_inputs,
        sampling_rate="",
        ):

    diarization = diarizer_pipeline({
        "waveform": diarizer_inputs,
        "sample_rate": sampling_rate
    })

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            'segment': {
                'start': segment.start,
                'end': segment.end
            },
            'track': track,
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
                "segment": {
                    "start": prev_segment["segment"]["start"],
                    "end": cur_segment["segment"]["start"]
                },
                "speaker": prev_segment["label"],
            })
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append({
        "segment": {
            "start": prev_segment["segment"]["start"],
            "end": cur_segment["segment"]["end"]
        },
        "speaker": prev_segment["label"],
    })

    return new_segments


def transcribe(
    asr_pipeline,
    inputs,
    segments,
    sampling_rate,
    group_by_speaker=True
):

    asr_out = asr_pipeline(
        {
            "array": inputs,
            "sampling_rate": sampling_rate
        },
        return_timestamps=True,
        generate_kwargs={
            "language": "<|ja|>",
            "task": "transcribe"
        }
    )
    transcript = asr_out["chunks"]

    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append({
                "speaker":
                segment["speaker"],
                "text":
                "".join([chunk["text"] for chunk in transcript[:upto_idx + 1]]),
                "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
            })
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

    return segmented_preds
