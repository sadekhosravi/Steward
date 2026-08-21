"""tau2's own CLI, with our agent registered first.

    uv run python scripts/run_bench.py run --domain mock --agent mas \
        --agent-llm openai/gpt-oss-20b --user-llm nvidia_nim/openai/gpt-oss-20b

Every `tau2 ...` command works here; the only difference is that `--agent mas`
resolves. On Windows set PYTHONUTF8=1 first.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # before tau2 imports: it reads its data directory at import time

# tau2's user simulator goes through LiteLLM, which looks for its own variable
# name. One key, two names, so .env only has to carry the one.
if "NVIDIA_API_KEY" in os.environ:
    os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ["NVIDIA_API_KEY"])

from tau2.cli import main  # noqa: E402

import adapters.tau2  # noqa: E402

if __name__ == "__main__":
    adapters.tau2.register()
    sys.exit(main())
