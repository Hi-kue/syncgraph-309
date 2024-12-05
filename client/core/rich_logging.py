import logging
import time

from rich.logging import RichHandler

start_time = time.time()  # time::now
logging.basicConfig(
    format="{asctime} - {levelname}: {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")
