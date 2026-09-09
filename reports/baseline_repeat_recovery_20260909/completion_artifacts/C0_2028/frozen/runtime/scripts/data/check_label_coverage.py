"""Compare actual recording/label names, excluding OS sidecars; no data mutation."""
import argparse
import json
import os
from pathlib import Path


def names(root, extension):
    clips, sidecars = set(), []
    if not root.is_dir():
        return clips, sidecars
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [name for name in dirs if not name.startswith('.')]
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() != extension:
                continue
            if name.startswith('.'):
                sidecars.append(str(path))
            else:
                clips.add(str(path.relative_to(root).with_suffix('')))
    return clips, sidecars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    audio, audio_sidecars = names(args.audio, '.wav')
    labels, label_sidecars = names(args.labels, '.csv')
    print(json.dumps(dict(audio_root=str(args.audio), label_root=str(args.labels),
                          audio=len(audio), labels=len(labels),
                          audio_without_labels=sorted(audio-labels), labels_without_audio=sorted(labels-audio),
                          sidecars=audio_sidecars+label_sidecars,
                          matched=bool(audio) and audio == labels), ensure_ascii=False))


if __name__ == '__main__':
    main()
