# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-whisper>=1.1"]
# ///
"""Transcribe generated WAVs with Whisper and compare them with the input text.

Input is a JSON list of {"wav": path, "text": expected, "language": "en"|"ja"}.
Reports the character error rate after removing punctuation, spaces and
bracketed tags such as [laughs] or <|speaker:0|>, which are not spoken.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", default="large-v3")
parser.add_argument("--device", default="cpu")
arguments = parser.parse_args()

TAG = re.compile(r"\[[^\]]*\]|<\|[^|]*\|>")


def normalize(text):
    text = unicodedata.normalize("NFKC", TAG.sub("", text)).lower()
    return "".join(character for character in text if unicodedata.category(character)[0] in "LN")


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, 1):
        current = [i]
        for j, right_character in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


model = WhisperModel(arguments.model, device=arguments.device, compute_type="int8" if arguments.device == "cpu" else "float16")
results = []
for case in json.loads(arguments.cases.read_text()):
    segments, info = model.transcribe(case["wav"], language=case.get("language"), beam_size=5)
    transcript = "".join(segment.text for segment in segments).strip()
    expected, heard = normalize(case["text"]), normalize(transcript)
    cer = edit_distance(expected, heard) / max(len(expected), 1)
    results.append({**case, "transcript": transcript, "cer": round(cer, 4), "audio_seconds": round(info.duration, 3)})
    print(f"{Path(case['wav']).name}: CER {cer:.3f} | {transcript}", flush=True)
arguments.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
