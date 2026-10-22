# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import logging
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level
    #format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
    format='%(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("text_normalization")
