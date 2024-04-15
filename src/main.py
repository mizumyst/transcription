# TODO: bad
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

import json
import pickle
from torch import save as torchsave

from pathlib import Path
from models import *


AUDIO_EXT = ('.wav', '.mp3')


def main(input_files,
         token=None,
         sampling_rate=16000,
         do_preprocess=True,
         do_diarize=True,
         do_transcribe=True,
         do_postprocess=True):

    # Create output path
    paths = []
    for filename in input_files:
        path = Path(f'_temp/{filename}')
        path.mkdir(parents=True, exist_ok=True)
        paths.append(path)


    current_pipeline = None

    if do_preprocess:
        print("Preprocessing ...")
        for filename, path in zip(input_files, paths):
            inputs, diarizer_inputs = preprocess(filename, sampling_rate=sampling_rate)

            with open(path / '0_inputs_asr', 'wb') as file:
                torchsave(inputs, file)

            with open(path / '0_inputs_sd', 'wb') as file:
                torchsave(diarizer_inputs, file)


    if do_diarize:
        print("Initializing diarization pipeline ...")
        current_pipeline = init_diarizer_pipeline(
            token=token
        )

        for path in tqdm(paths, desc="Diarizing ..."):
            diarizer_inputs = torch.load(path / '0_inputs_sd')

            segments = diarize(
                current_pipeline,
                diarizer_inputs,
                sampling_rate=sampling_rate
            )

            with open(path / '1_segments', 'wb') as file:
                pickle.dump(segments, file, protocol=pickle.HIGHEST_PROTOCOL)

        reset_cuda(current_pipeline)


    if do_transcribe:
        print("Initializing ASR pipeline ...")
        current_pipeline = init_asr_pipeline(
            token=token
        )

        for path in tqdm(paths, desc="Transcribing ..."):
            inputs = torch.load(path /  '0_inputs_asr')
            with open(path / '1_segments', 'rb') as file:
                segments = pickle.load(file)

            preds = transcribe(current_pipeline, inputs, segments, sampling_rate)
            with open(path / '2_preds', 'wb') as file:
                pickle.dump(preds, file, protocol=pickle.HIGHEST_PROTOCOL)

        reset_cuda(current_pipeline)
    

    if do_postprocess:
        print("Postprocessing ...")
        for filename, path in zip(input_files, paths):
            with open(path / '2_preds', 'rb') as file:
                result = pickle.load(file)

            with open(f'{filename}.json', 'w', encoding='utf-8') as file:
                for chunk in result:
                    chunk['speaker'] = int(chunk['speaker'][-2:])
                    chunk['start'], chunk['end'] = chunk['timestamp']
                    del chunk['timestamp']
                json.dump(result, file)


if __name__ == '__main__':
    data_path = Path('data')
    input_paths = [str(path) for path in data_path.iterdir() if path.suffix in AUDIO_EXT]

    print(input_paths)

    main(
        input_paths,
        token="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"
    )
