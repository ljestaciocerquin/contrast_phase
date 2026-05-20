import torch
import os
import pandas as pd

model_path = "/projects/contrast_phase/Diffusion/trained_LDDM_model.pth"

model=torch.load(model_path, map_location=torch.device("cpu"))
