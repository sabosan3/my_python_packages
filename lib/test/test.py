# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 03:56:32 2018

@author: Taiki Sato
"""

import numpy as np
import time
from matrix import *

def original_rank():
    
    mat = np.arange(1000000)
    mat = mat.reshape((400, -1))
    
    start = time.time()
    util.rank(mat)
    end = time.time()
    
    rank = util.rank(mat)
    
    print("time = %s" % (end - start))
    print("rank = %s" % rank)
    
def np_rank():
    
    mat = np.arange(1000000)
    mat = mat.reshape((400, -1))
    
    start = time.time()
    np.linalg.matrix_rank(mat)
    end = time.time()
    
    rank = np.rank(mat)
    
    print("time = %s" % (end - start))
    print("rank = %s" % rank)