import logging
import os
import cv2
import time
from soph import logs_path

def initiate_logging(log_name):
    timestr = time.strftime("%Y-%m-%d-%H-%M")
    log_dir = os.path.join(logs_path, timestr)
    os.mkdir(log_dir)
    filename = os.path.join(log_dir, log_name)
    logging.root.handlers = []
    logging.basicConfig(
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(filename),
                            logging.StreamHandler()
                        ])
    return log_dir

def save_map(log_dir, map):
    filename = time.strftime("%H-%M-%S-map.png")
    map_cv = cv2.cvtColor(map*255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(log_dir, filename), map_cv)
    logging.info("Saved Occupancy Map as " + filename)