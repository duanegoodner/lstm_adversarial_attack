import logging
import lstm_adversarial_attack.config_paths as cfp
from pathlib import Path

# Create loggers for two different parts of your application
logger_a = logging.getLogger('logger_a')
logger_b = logging.getLogger('logger_b')

# Configure handlers for each logger
handler_a = logging.FileHandler(cfp.HYPERPARAMETER_OUTPUT_DIR / 'log_file_a.csv')
handler_b = logging.FileHandler(cfp.HYPERPARAMETER_OUTPUT_DIR / 'log_file_b.csv')

# Create formatters for the log messages
formatter = logging.Formatter('%(asctime)s')
formatter.default_msec_format = '%s.%03d'
handler_a.setFormatter(formatter)
handler_b.setFormatter(formatter)

# Add handlers to the loggers
logger_a.addHandler(handler_a)
logger_b.addHandler(handler_b)

# Set logging levels for the loggers
logger_a.setLevel(logging.INFO)
logger_b.setLevel(logging.DEBUG)

# Example log messages
logger_a.info('This message goes to log file A')
# logger_b.debug('This message goes to log file B')
