from joblib import Parallel, delayed
from neuromaps.stats import compare_images
import numpy as np

def compare_geomodes_parallel(subjects_array, num_modes=200, n_jobs=1):
    num_subjects = subjects_array.shape[0]
    
    jobs = []
    
    for subi in range(num_subjects):
        subject1 = subjects_array[subi]
        
        for subj in range(subi+1, num_subjects):
            subject2 = subjects_array[subj]
            
            jobs.append(delayed(compare_modes)(
                subject1, subject2, num_modes)
                )
            
    corr = np.vstack(
        Parallel(n_jobs=n_jobs, prefer='threads')(jobs)
        )
    
    return np.asarray(corr).squeeze()

def compare_modes(subject1, subject2, num_modes):
    corr = np.zeros(num_modes)
    for j in range(num_modes):
        corr[j] = compare_images(subject1[j], subject2[j], metric='pearsonr')

    return np.asarray(corr).squeeze()