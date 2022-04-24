import sys
from pathlib import Path
from time import perf_counter as timer

import numpy as np
import torch

from voiceapi.voiceapi.voiceclonelib.encoder import inference as encoder
from voiceapi.voiceapi.voiceclonelib.synthesizer import Synthesizer
from voiceapi.voiceapi.voiceclonelib.toolbox.utterance import Utterance
from voiceapi.voiceapi.voiceclonelib.vocoder import inference as vocoder
import os


# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

# Maximum of generated wavs to keep on memory
MAX_WAVS = 15


class Toolbox:
    def __init__(self, datasets_root: Path, models_dir: Path, seed: int=None):
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.models_dir = models_dir
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav

        self.synthesizer = None # type: Synthesizer
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []
        self.seed = seed

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False


    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]


    def load_from_browser(self, fpath=None):
        name = fpath.name
        speaker_name = fpath.parent.name

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.add_real_utterance(wav, name, speaker_name)


    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)

    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self, texts):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or self.seed is not None:
            self.init_synthesizer()

        texts = texts.split("\n")
        embed = self.ui.selected_utterance.embed
        embeds = [embed] * len(texts)
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)


    def vocode(self, name):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        seed = None

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
        #if self.ui.current_vocoder_fpath is not None:
        wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        #else:
        #wav = Synthesizer.griffin_lim(spec)

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        wav = encoder.preprocess_wav(wav)
        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        return wav


        #Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVS:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        # Update current wav
        self.set_current_wav(0)
        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)


    def init_encoder(self):
        model_fpath = os.path.join(self.model_path, "encoder.pt")
        start = timer()
        encoder.load_model(model_fpath)


    def init_synthesizer(self):
        model_fpath = os.path.join(self.model_path, "synthesizer.pt")
        start = timer()
        self.synthesizer = Synthesizer(model_fpath)


    def init_vocoder(self):
        model_fpath = os.path.join(self.model_path, "vocoder.pt")
        # Case of Griffin-lim
        if model_fpath is None:
            return
        start = timer()
        vocoder.load_model(model_fpath)

