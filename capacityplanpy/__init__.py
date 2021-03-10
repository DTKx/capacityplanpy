import os, logging

logging.basicConfig(
    filename=os.path.join(os.getcwd(), "capacityplan.log"), filemode="w", level=logging.ERROR
)  # Defines the path and level of log file
