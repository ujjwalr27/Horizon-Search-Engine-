
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Setup centralized logging with rotating file handler
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'horizon_search_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # Rotating file handler
            RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        ]
    )
    
    return logging.getLogger(__name__)

# Configure loggers for specific modules
def get_module_logger(module_name):
    """
    Get a module-specific logger
    """
    return logging.getLogger(module_name)
