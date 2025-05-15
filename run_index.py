import warnings
warnings.filterwarnings('ignore')
from prepare_db import PaperSemanticVectorIndex
from omegaconf import OmegaConf
from hydra import initialize, compose

with initialize(config_path="conf"):
    cfg = compose(config_name="index_config")

'''
This script is used to create a vector index for the paper embeddings.

Can include inference routines here or use run_inference.py
'''

db = PaperSemanticVectorIndex(cfg)