# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 23:36:51 2018

@author: Taiki Sato
"""

from typing import List, Union, Any, Dict
import numpy as np
import copy

def rank(mat: Any, err: int = 6) -> int:
    
    if is_all_elements_zero(mat):    
        print("All elements are Zero")
        return mat
    
    for m, _, _, _, ret in gaussian_elimination(mat, err=err):
        if ret: gaussian_elimination_mat = m

    gaussian_elimination_mat_sum = gaussian_elimination_mat.sum(axis=1)
    
    rank = 0
    for i in gaussian_elimination_mat_sum:
        if i == 0:
            break
        rank += 1
        
    return rank

def gaussian_elimination(mat: Any, err: int = 6) -> Any:

    if is_all_elements_zero(mat, err=err):    
        print("All elements are Zero")
        return mat

    mat = copy.deepcopy(mat)
    
    if mat.shape[0] > mat.shape[1]:
        mat = mat.T
    
    row, col = [0, 0]
    
    while row <= mat.shape[0]-1:
        
        if is_all_elements_zero(mat[row:, col:], err=err):
            break
        
        mat, p, q = exchange_abs_max_value(mat, [row, col])

        identity_mat = np.eye(mat.shape[0], mat.shape[0])
        identity_mat[row + 1:, col] = -1 * (mat[row + 1:, col] / mat[row, col])
       
        mat = np.dot(identity_mat, mat)
        
        row += 1
        col += 1

        yield mat, identity_mat, p, q, False
        
    mat = mat.round(err)
                
    yield mat, None, None, None, True

def is_all_elements_zero(mat: Any, err: int = 6) -> bool:
    
    tmp_mat = mat.round(err)
    
    if np.abs(tmp_mat).sum() == 0:
        return True
    else:
        return False

def search_abs_max_pos(mat: Any) -> List:
    
    abs_max_row, abs_max_col = np.where(np.abs(mat)==np.abs(mat).max())
    abs_max_row = abs_max_row[0]
    abs_max_col = abs_max_col[0]
    
    return [abs_max_row, abs_max_col]
    
def exchange_abs_max_value(mat: Any, pos: List = [0, 0]) -> Any:

    row, col = pos
    target_mat = mat[row:, col:]
    abs_max_row, abs_max_col = search_abs_max_pos(target_mat)
    
    p = np.eye(mat.shape[0])
    q = np.eye(mat.shape[1])
    
    _, p_tmp = exchange_row(target_mat, 0, abs_max_row)
    _, q_tmp = exchange_col(target_mat, 0, abs_max_col)

    p[row:, row:] = p_tmp
    q[col:, col:] = q_tmp
    
    return np.dot(np.dot(p, mat), q), p, q
    
def exchange_row(mat: Any, row1: int, row2: int) -> Any:
        
    i = np.eye(mat.shape[0])
    
    tmp_row = copy.deepcopy(i[row1,:])
    i[row1,:] = i[row2,:]
    i[row2,:] = tmp_row

    return np.dot(i, mat), i

def exchange_col(mat: Any, col1: int, col2: int) -> Any:
    
    i = np.eye(mat.shape[1])
    
    tmp_col = copy.deepcopy(i[:, col1])
    i[:, col1] = i[:, col2]
    i[:, col2] = tmp_col
    
    return np.dot(mat, i), i