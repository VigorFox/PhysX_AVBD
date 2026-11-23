"""Recompute the published statistics from the archived Checked runs."""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    fixture = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        shutil.copyfile(fixture / "registration.json", work / "registration.json")
        with gzip.open(fixture / "study.json.gz", "rb") as source:
            (work / "study.json").write_bytes(source.read())
        with gzip.open(fixture / "raw.json.gz", "rt", encoding="utf-8") as source:
            records = json.load(source)
        (work / "tail").mkdir()
        for arm in ("atomic", "batch"):
            for workers in (4, 8):
                for repeat in range(3):
                    key = f"tail/volume-{workers}-{repeat}-{arm}.json"
                    (work / key).write_text(records[key], encoding="utf-8")
        subprocess.run([sys.executable, fixture / "analyze.py", work], check=True)
        actual = json.loads((work / "analysis.json").read_bytes())
        expected = json.loads((fixture / "analysis.json").read_bytes())
        if actual != expected:
            sys.exit("Archived statistics differ from replay")
        print("All reported statistics match the archived runs")


if __name__ == "__main__":
    main()
