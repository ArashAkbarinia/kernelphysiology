"""
The main function for the export_weights_to_model.sh.
"""

import sys

from kernelphysiology.dl.keras.models.utils import export_weights_to_model

weights_path = sys.argv[1]
model_path = sys.argv[2]
architecture = sys.argv[3]
dataset = sys.argv[4]
export_weights_to_model(weights_path, model_path, architecture, dataset,
                        area1layers=None)
