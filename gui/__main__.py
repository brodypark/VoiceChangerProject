"""GUI entry point: `python -m gui [same options as main.py]`.

Command-line options and .env set the starting values; options without a control in the
window (--timing, --tts-model, --delivery, --output-format, --denoise, --vad) apply as given.
"""

from __future__ import annotations

import signal
import sys
from typing import Sequence

from PyQt6.QtWidgets import QApplication

from config import ConfigError, load_config
from gui.window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = load_config(sys.argv[1:] if argv is None else argv)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    app = QApplication(sys.argv[:1])
    window = MainWindow(config)
    # Ctrl+C in the terminal closes the window cleanly. The window's 100 ms timer runs Python code,
    # which gives the interpreter a chance to handle the signal while Qt's event loop is running.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
