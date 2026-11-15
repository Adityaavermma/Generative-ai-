"""Start the Streamlit UI from any folder:  python run_app.py [--server.port 8501]

Streamlit reads .streamlit/config.toml (theme) from the current directory, so
this switches to the project folder first.
"""

import os
import sys
from pathlib import Path

from streamlit.web import cli

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(HERE)
    sys.argv = ["streamlit", "run", str(HERE / "app.py"), *sys.argv[1:]]
    sys.exit(cli.main())
