import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

fixture = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as scratch:
    work = Path(scratch)
    shutil.copyfile(fixture / "registration.json", work / "registration.json")
    with gzip.open(fixture / "runs.json.gz", "rb") as stream:
        (work / "runs.json").write_bytes(stream.read())
    subprocess.run([sys.executable, fixture / "analyze.py", work], check=True)
    if json.loads((work / "analysis.json").read_bytes()) != json.loads((fixture / "analysis.json").read_bytes()):
        sys.exit("Paired statistics differ from the archived measurements")
    print("All paired statistics reproduced")
