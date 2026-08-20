"""Fetch the tau2 benchmark data that the installed `tau2` library expects.

tau2 resolves its data directory as `<repo root>/data` (see tau2.utils.utils),
which does not exist when tau2 is installed as a library. The supported escape
hatch is the TAU2_DATA_DIR environment variable, so this script materialises a
minimal checkout of that directory and prints the value to point at it.

Only the domain definitions and user-simulator prompts are checked out; the bulk
of upstream `data/` is published leaderboard results (~577M) that we never read.
The revision is taken from pyproject.toml so the data always matches the pinned
library.

Usage:
    python scripts/bootstrap_data.py
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "vendor" / "tau2-data"

# Sparse paths, relative to the upstream repo root.
SPARSE_PATHS = [
    "data/tau2/domains",
    "data/tau2/user_simulator",
]


def read_pinned_source() -> tuple[str, str]:
    """Return (git_url, rev) for tau2 from pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    try:
        source = pyproject["tool"]["uv"]["sources"]["tau2"]
        return source["git"], source["rev"]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise SystemExit(
            "pyproject.toml is missing [tool.uv.sources].tau2 with 'git' and 'rev' keys"
        ) from exc


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> int:
    url, rev = read_pinned_source()

    if not (TARGET / ".git").exists():
        print(f"Cloning {url} (blobless, sparse) into {TARGET}")
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", "--sparse", url, str(TARGET))
        git("sparse-checkout", "set", "--no-cone", *SPARSE_PATHS, cwd=TARGET)
    else:
        print(f"Updating existing checkout in {TARGET}")
        git("fetch", "--filter=blob:none", "origin", cwd=TARGET)

    print(f"Checking out {rev}")
    git("checkout", "--detach", rev, cwd=TARGET)

    data_dir = TARGET / "data"
    mock = data_dir / "tau2" / "domains" / "mock"
    if not mock.is_dir():
        print(f"error: expected domain data at {mock}", file=sys.stderr)
        return 1

    domains = sorted(p.name for p in (data_dir / "tau2" / "domains").iterdir() if p.is_dir())
    print(f"\nOK. Domains available: {', '.join(domains)}")
    print(f"Set TAU2_DATA_DIR={data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
