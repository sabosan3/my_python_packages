# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:20:59 2018

@author: Taiki Sato
"""

from typing import List, Union, Any, Dict
import sys
import numpy as np

from matrix import util

def LU(mat: Any, LDU: bool = False, err: int = 6) -> Any:

    try:    
        if mat is None or mat.size == 0:
            print("Invalid argments are inputted")
            return False
        
        if util.is_all_elements_zero(mat):
            print("All elements are Zero")
            return False
        
        if mat.shape[0] > mat.shape[1]:
            transposed = True
        else:
            transposed = False
            
        rank = util.rank(mat)
        
        for iter_num, (m, i, p, _, ret) in enumerate(util.gaussian_elimination(mat)):
            if ret:
                gauss_elimi_mat = m
                continue
            
            if iter_num == 0:
                l = i
            else:
                l = np.dot(i, np.dot(p, np.dot(l, p.T)))

        diag = gauss_elimi_mat.diagonal()

        tmp = np.reshape(diag[:rank], (-1,1))
        tmp = gauss_elimi_mat[:rank, :]/np.repeat(tmp, gauss_elimi_mat.shape[1], axis=1)
        u = np.eye(gauss_elimi_mat.shape[1])
        u[:rank,:] = tmp
        u = u.round(err)

        diag_mat = np.diag(diag)
        d = np.zeros_like(gauss_elimi_mat)
        d[:, :diag_mat.shape[1]] = diag_mat
        d = d.round(err)

        l = np.linalg.inv(l)
        l = l.round(err)
      
        if LDU:
            if transposed:
                return u.T, d.T, l.T
            else:
                return l, d, u
        else:
            if transposed:
                return u.T, np.dot(d.T, l.T)
            else:
                return l, np.dot(d, u)  
        
    except Exception as e:
        print("Occur exception : %s" % e)
        
def QR(mat: Any) -> Any:
    
    return True

def SVD(mat: Any) -> Any:
    
    return True

