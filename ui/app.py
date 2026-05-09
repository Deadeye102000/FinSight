"""Top-level Streamlit entrypoint for the FinSight dashboard."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finsight.ui.app import main


if __name__ == "__main__":
    main()
