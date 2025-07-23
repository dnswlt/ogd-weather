# Global logging config
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
