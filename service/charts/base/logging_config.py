"""Global logging config.

Import this module once per application (in its "main" module) as

from service.charts.base import logging_config as _  # configure logging

This ensures that logging is configured consistently.
"""

import logging
import os

logging.basicConfig(
    level=(
        logging.DEBUG
        if os.environ.get("OGD_LOG_LEVEL", "INFO").upper() == "DEBUG"
        else logging.INFO
    ),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
