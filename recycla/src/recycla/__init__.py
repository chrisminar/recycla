import logging
import os
import sys
from pathlib import Path
from typing import Final

import torch

VERSION = "0.4.5"

# Use environment variables if available, otherwise fall back to relative paths
ROOT_PATH: Final[Path] = Path(
    os.environ.get(
        "RECYCLA_ROOT_PATH", Path(__file__).resolve().parent.parent.parent.parent
    )
)
DATA_PATH: Final[Path] = ROOT_PATH / "data"
LOCAL_CAPTURE_PATH: Final[Path] = DATA_PATH / "local_capture"
CONFIG_PATH: Final[Path] = ROOT_PATH / "recycla/src/recycla/config"

# Configure logging more robustly
# Clear any existing handlers to start fresh
logging.getLogger().handlers.clear()

# Create a console handler for your recycla logs only
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a formatter for better log output
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)


# Force the handler to flush immediately for shell redirection
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# Use stderr instead of stdout to avoid conflicts with tee
console_handler = FlushingStreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Set up the root logger with WARNING level to suppress third-party logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

# Explicitly silence common noisy third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torchvision").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Set up your recycla package logger to show all your logs
recycla_logger = logging.getLogger("recycla")
recycla_logger.setLevel(logging.DEBUG)
recycla_logger.addHandler(console_handler)

# Force unbuffered output for shell redirection compatibility
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    # Fallback for older Python versions or when stdout is not a terminal
    pass

# Only disable propagation if not in testing mode
# This allows pytest's caplog to capture logs during testing
if "pytest" not in sys.modules:
    recycla_logger.propagate = False  # Don't propagate to root logger in production

# get simple handle to recycla logger
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SERVICE_ACCOUNT_KEY: Final[Path] = ROOT_PATH / ".secrets/service_account_key.json"
PI_ID_FILE_PATH: Final[Path] = ROOT_PATH / ".secrets/piauth.json"
PI_ENV: Final[str] = "PI_ID"
DEFAULT_BUCKET_NAME: Final[str] = "recyclo-c0fd1.firebasestorage.app"
