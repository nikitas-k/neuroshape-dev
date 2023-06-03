from os import path, system
import nibabel as nib

def load_mesh_vertices(vx):
    """
    Loads mesh vertices given by gifti surface filename 'vx'
    
    Returns
    -------
    vertices : (N,3) np.ndarray
    Coordinates of surface vertices
    
    """

    mesh = nib.load(vx)
    vertices = mesh.agg_data('pointset')
    
    return vertices