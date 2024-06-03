#!/usr/bin/env python3

from os import getenv
import os
import logging
import sys

class ServiceConfig(object):
    LOG_LEVEL='DEBUG'



def setup_logging(default_level=logging.INFO):
    """
    | Logging Setup
    for more info https://gist.github.com/kingspp/9451566a5555fb022215ca2b7b802f19
    """
    if ServiceConfig.LOG_LEVEL == "DEBUG":
        default_level = logging.DEBUG
    elif ServiceConfig.LOG_LEVEL == "INFO":
        default_level = logging.INFO
    elif ServiceConfig.LOG_LEVEL == "WARNING":
        default_level = logging.WARNING

    logging.basicConfig(
        stream=sys.stdout,
        level=default_level,
        format="[%(asctime)s] {%(process)d %(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    # check if module exists
    try:
        import coloredlogs

        coloredlogs.install(
            level=default_level,
            fmt="[%(asctime)s,%(msecs)03d] {%(process)d %(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)
    except ImportError:
        logger.warning(
            f"coloredlogs module does not exists so skipping coloredlogs logging setup"
        )
    logger.info(f"Setting Logging Config with Level: {default_level}")
