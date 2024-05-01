import click
from tqdm import tqdm

import json
import pickle
from torch import load as torchload, save as torchsave
import pandas as pd

import soundfile as sf

from pathlib import Path
import models
import configparser


AUDIO_EXT = ('.wav', '.mp3')


def preprocess(input_files, paths, sampling_rate, force=False):
    if not force and all(
        (path / '0_inputs_asr').is_file() and (path / '0_inputs_sd').is_file()
        for path in paths
    ):
        return
    
    for filename, path in tqdm(zip(input_files, paths), desc="Preprocessing ..."):
        if not force and (path / '0_inputs_asr').is_file() and (path / '0_inputs_sd').is_file():
            continue

        inputs, diarizer_inputs = models.preprocess(filename, sampling_rate=sampling_rate)

        with open(path / '0_inputs_asr', 'wb') as file:
            torchsave(inputs, file)

        with open(path / '0_inputs_sd', 'wb') as file:
            torchsave(diarizer_inputs, file)


def diarize(paths, token, sampling_rate, force=False):
    if not force and all((path / '1_segments').is_file() for path in paths):
        return

    tqdm.write("Initializing diarization model ...")
    current_model = models.init_diarizer(token=token)

    for path in tqdm(paths, desc="Diarizing ..."):
        if not force and (path / '1_segments').is_file():
            continue

        diarizer_inputs = torchload(path / '0_inputs_sd')

        segments = models.diarize(
            current_model,
            diarizer_inputs,
            sampling_rate=sampling_rate
        )

        with open(path / '1_segments', 'wb') as file:
            pickle.dump(segments, file, protocol=pickle.HIGHEST_PROTOCOL)

    models.reset_cuda(current_model)


def transcribe(paths, do_translate=False, force=False):
    if not force and all(
        (path / '2_preds').is_file() and (path / '2_speakers').is_file()
        for path in paths
    ):
        return

    tqdm.write("Initializing ASR model ...")
    current_model = models.init_whisper()

    for path in tqdm(paths, desc="Transcribing ..."):
        if not force and (path / '2_preds').is_file() and (path / '2_speakers').is_file():
            continue

        inputs = torchload(path /  '0_inputs_asr')
        with open(path / '1_segments', 'rb') as file:
            segments = pickle.load(file)

        _, preds, speaker_counts = models.transcribe(
            current_model,
            inputs, segments,
            count=True
        )
        speaker_codes = list(zip(*speaker_counts))[0]

        with open(path / '2_preds', 'wb') as file:
            pickle.dump(preds, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / '2_speakers', 'wb') as file:
            pickle.dump(speaker_codes, file, protocol=pickle.HIGHEST_PROTOCOL)

        if not do_translate:
            continue

        text, preds_tr = models.transcribe(
            current_model,
            inputs, segments,
            task='translate'
        )

        with open(path / '2_preds_tr', 'wb') as file:
            pickle.dump(preds_tr, file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / '2_text_tr', 'w') as file:
            file.write(text)

    models.reset_cuda(current_model)


def postprocess(input_files, paths, metadata, do_translate=False):
    tqdm.write("Postprocessing ...")

    for input_path, path in zip(input_files, paths):
        input_file = path.name
        if input_file not in metadata:
            continue

        current_metadata = metadata[input_file]

        with open(path / '2_preds', 'rb') as file:
            preds = pickle.load(file)

        if do_translate:
            with open(path / '2_text_tr', 'r') as file:
                text_tr = file.read()

        # Convert speakers to provided names
        if current_metadata['speakers']:
            current_metadata['speakers'] = current_metadata['speakers'].split()

            with open(path / '2_speakers', 'rb') as file:
                speaker_codes = pickle.load(file)
            assert len(speaker_codes) == len(current_metadata['speakers']), f"{speaker_codes} vs {current_metadata}"

            codes = {code: name for code, name in zip(speaker_codes, current_metadata['speakers'])}

            for pred in preds:
                pred['speaker'] = codes[pred['speaker']]

        f = sf.SoundFile(input_path)
        time = (len(f) * 1000) // f.samplerate 
        current_metadata['time'] = time

        with open(f'{input_path}.json', 'w', encoding='utf-8') as file:
            result = {
                'metadata': current_metadata,
                'transcribe': preds
            }
            if do_translate:
                result['translate'] = text_tr
            
            json.dump(result, file, indent=4)
        
        with open(f'{input_path}.txt', 'w', encoding='utf-8') as file:
            file.write(text_tr)


def main(input_files, metadata, token, sampling_rate=16000, do_translate=True):
    # Create output path
    paths = []
    for filename in input_files:
        path = Path(f'_temp/{filename}')
        path.mkdir(parents=True, exist_ok=True)
        paths.append(path)

    preprocess(input_files, paths, sampling_rate)
    diarize(paths, token=token, sampling_rate=sampling_rate)
    transcribe(paths, do_translate)
    postprocess(input_files, paths, metadata, do_translate)


class ProgressHook:
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
        self.progress.close()

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
            self.progress[step_name] = tqdm(total=total)

        self.progress.update(completed)

        # force refresh when completed
        if completed >= total:
            self.progress.refresh()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('settings.ini')

    data_path = Path('data')

    tqdm.write(f"Data path: {data_path}")
    manifest_path = None
    input_paths = []
    input_filenames = set()
    for path in data_path.iterdir():
        if path.suffix in AUDIO_EXT:
            input_paths.append(str(path))
            input_filenames.add(path.name)
        elif path.stem == 'manifest':
            manifest_path = str(path)
    
    if not manifest_path:
        tqdm.write("ERROR: Could not find manifest file.")
        exit()

    manifest = pd.read_csv(manifest_path).set_index('filename').to_dict(orient='index')
    a, b = set(input_filenames), set(manifest.keys())
    manifest_only, folder_only = a - b, b - a
    mismatch = manifest_only | folder_only
    if mismatch:
        tqdm.write("\nMismatch detected.")
        if manifest_only:
            tqdm.write(f"\tFiles missing from manifest: {manifest_only}")
        if folder_only:
            tqdm.write(f"\tFiles missing from folder: {folder_only}")
        
        if not click.confirm("\nContinue anyways?", default=True):
            exit()
        
        new_manifest = dict()
        for key, item in manifest.items():
            if key not in mismatch:
                new_manifest[key] = item
        
        manifest = new_manifest

    main(
        input_paths,
        manifest,
        token=config['token']
    )
