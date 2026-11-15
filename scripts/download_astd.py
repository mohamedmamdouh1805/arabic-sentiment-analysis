import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/Tweets.txt"
SHA256 = "616de26a4e420d98611e16a2c4b61be6b83fd6e6951a8a4985da565871506e04"
MAX_BYTES = 2_000_000


def main():
    parser = argparse.ArgumentParser(description="Download and verify the 1.49 MB ASTD data file.")
    parser.add_argument("output", type=Path)
    path = parser.parse_args().output
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
            raise SystemExit(f"Existing file has an unexpected SHA-256: {path}")
        print(f"Verified existing dataset: {path}")
        return
    with urlopen(URL, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("Download size or SHA-256 verification failed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
    print(f"Saved {len(data):,} bytes to {path}")


if __name__ == "__main__":
    main()
