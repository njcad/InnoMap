"""
Main file to build and train Graph Convolutional Network.
"""

import pandas as pd 
import numpy as np 



def construct_graph(path):
    """
    Load in cleaned data from path and, for each entry,
        1. embed abstract as node feature vector
        2. construct edge list from id to reference ids
    Then convert this to a PyTorch Geometric data object.
    """
    pass

