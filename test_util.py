import numpy as np
import math as math


    
"""
Report support recovery performance
"""
    
def check_support(Theta_truth, Theta_rec, p):
    mask_truth_bool = np.abs(Theta_truth)>1e-3
    mask_truth = mask_truth_bool.astype(int)
    mask_rec_bool = np.abs(Theta_rec)>1e-3
    mask_rec = mask_rec_bool.astype(int)
    
    nonzeors_true = np.sum(mask_truth)
    nonzeors_rec = np.sum(mask_rec)
    
    zeros_rec = p*p - nonzeors_rec

    
    fn_mask_bool = (mask_truth - mask_rec)>1e-4
    fn_mask = fn_mask_bool.astype(int)
    fp_mask_bool = (mask_truth - mask_rec)<-1e-4
    fp_mask = fp_mask_bool.astype(int)
    
    FP = np.sum(fp_mask)
    FN = np.sum(fn_mask)
    TP = nonzeors_rec - FP
    TN = zeros_rec - FN
    
    
    a = np.sqrt((TP+FP)*(TP+FN))
    b = np.sqrt((TN+FP)*(TN+FN))
    
    MCC = (TP*TN-FP*FN)/a/b
    
    return FP, FN, nonzeors_rec, nonzeors_true, MCC