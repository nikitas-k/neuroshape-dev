# import libraries
import pyvista as pv
import numpy as np
import scipy.sparse as sparse
from skimage.filters._gaussian import gaussian
from argparse import ArgumentParser
import sys
import os
        
def m_print(message):
    """
    print message, then flush stdout
    """
    print(message)
    sys.stdout.flush()


"""
RFT.py - Implementation of RFT on surface adjacency matrices to find clusters 
at specified statistical alpha. 

@author: Nikitas C. Koussis, Systems Neuroscience Group, University of Newcastle

Read the help file for information. Usage:

    $ python rft.py -h

Inputs: 
    
    <surface>                - precalculated association map with structure
                               <v_index> <x> <y> <z> <value>

    <q>                      - statistical alpha

    <outfile>                - path to output directory

Optional inputs:
    
    <--alpha> <float>        - Delaunay triangulation alpha, parameter for 
                               defining extent of convex hull, default 5
                        
    <--n_iter> <int>         - Number of smoothing iterations for geometry 
                               smooothing, default 200
    
    <--relax_factor> <float> - Relaxation factor (how much neighboring faces
                               contribute to each face), default 0.01 
    
    <--sigma> <float>        - Variance of Gaussian kernel to apply to adjacency
                               matrix. Default 1.
    
    <-Z> <float>             - Maximum Z threshold to estimate EC. Default 50.0
    
Optional outputs:
    
    <--output_thr>           - output Z threshold to stdout, default True
    
    <--ec>                   - output Euler characteristic for threshold,
                               default True
    
    <--resel>                - output resel count at `sigma`, default True

Example usage:
    $ python rft.py surface.asc 0.05 /home/user/ \
        --alpha 5 --n_iter 200 \
        --relax_factor 0.01 \
        --output_thr \
        --output_ec \
        --output_resel \
        --sigma 1.0
    
"""

def _filter(img, sigma):
    return gaussian(img, sigma, mode='wrap')

def construct_adj_sym(data, v, t):
    """
    Generate sparse edge (adjacency) matrix weighted by values in `data`
    of surface with vertices `v` and faces `t`.

    Parameters
    ----------
    data : np.ndarray of floats of shape (N,)
        Scalar values for each vertex in `v`.
    v : np.ndarray of floats of shape (N, 3)
        Vertex coordinates
    t : np.ndarray of ints of shape (M, 3)
        Triangles for each vertex, labeled by vertex index

    Returns
    -------
    scipy.sparse.csc_matrix of shape (N, N)

    """
    
    t0 = t[:, 0]
    t1 = t[:, 1]
    t2 = t[:, 2]
    w01 = w12 = w20 = map_vertex_func(data, t)
    i = np.column_stack((t0, t1, t1, t2, t2, t0)).reshape(-1)
    j = np.column_stack((t1, t0, t2, t1, t0, t2)).reshape(-1)
    dat = np.column_stack((w01, w12, w12, w20, w20, w01)).reshape(-1)
    n = v.shape[0]
    
    return sparse.csc_matrix((dat, (i, j)), shape=(n, n)).maximum(sparse.csc_matrix((dat, (j, i)), shape=(n, n))).todense()


def map_vertex_func(data, t):
    """
    Map vertex values in `data` to triangles by dividing each value by 3

    Parameters
    ----------
    data : np.ndarray of floats of shape (N,)
        Scalar values for each vertex
    t : np.ndarray of ints of shape (M, 3)
        Triangles for each vertex, labeled by vertex index

    Returns
    -------
    face_values : np.ndarray of floats of shape (M,)
        Triangular weights

    """
    vertex_func = np.array(data) / 3.0
    tria_func = np.sum(vertex_func[t], axis=1)    
    
    return tria_func


def _expected_ec(z, resel_count):
    return (resel_count * (4 * np.log(2)) * ((2 * np.pi) ** (-3./2)) * z) * np.exp((z ** 2) * (-.5))

def euler_threshold(adj, q=0.05, sigma=1.0, max_z=50.0):
    """
    Threshold by solution of expected Euler characteristic on (smoothed)
    edge adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray of shape (N, N)
        Edge adjacency matrix with weights derived from vertex-wise data.
    q : float, optional
        Statistical alpha for significance. Default 0.05
    sigma : float, optional
        Number of standard deviations of Gaussian filter to apply (i.e., the 
        width of the filter). Default is 1.0
    max_z : float, optional
        Maximum threshold to compute EC to. Default is 50.0

    Returns
    -------
    float
        Z (threshold) where expected Euler characteristic is less than or 
        equal to the statistical alpha, (ideally) minimizing FDR.

    """
    g = _filter(adj, sigma=sigma)
    resels = np.prod(g.shape) / (2.355 * sigma)
    # estimate Z
    z = np.linspace(0, max_z, 10000)
    ec = _expected_ec(z, resels)
    if z[np.argwhere(ec <= q)[1]] is Exception:
        raise RuntimeError("Unable to compute EC, increase maximum threshold")
        
    return float(z[np.argwhere(ec <= q)[1]]), ec, resels


def make_surface(surface_file, alpha=5., n_iter=200, relax_factor=0.01):
    """
    Make triangular surface based on vertex coordinates in `surface_file`.

    Parameters
    ----------
    surface_file : str
        Path to surface file in .asc format. Expects structure to be set out as
        follows (no header):
            
            <vertex_index_1> <x_1> <y_1> <z_1> <scalar_1>
            <vertex_index_2> <x_2> <y_2> <z_2> <scalar_2>
            ...
            
        Future implementations will receive input from most neuroimaging formats.
    alpha : float, optional
        Delaunay triangulation smoothing. Decreases the sharpness of the
        edges of the triangles, but also increases the deformation of the convex 
        hull (makes it larger). Higher or lower values may be needed when 
        degenerate triangles (holes) are present. Default is 5.0.
    n_iter : int, optional
        Number of surface smoothing iterations. Default is 200.
    relax_factor : float, optional
        Triangulation relaxation factor. Defines how smooth local neighborhood 
        geometry is. Default is 0.01.

    Returns
    -------
    mesh :
        tuple of (np.ndarray of vertices, np.ndarray of faces)

    """
    
    surface = np.loadtxt(surface_file)
    verts = surface[:, 1:4]

    # convert point cloud in surface
    cloud = pv.PolyData(verts)
    volume = cloud.delaunay_3d(alpha=alpha)
    shell = volume.extract_geometry()
    smooth = shell.smooth(n_iter=n_iter, relaxation_factor=relax_factor,
                          feature_smoothing=False, 
                          boundary_smoothing=True,
                          edge_angle=100, feature_angle=100)

    # extract faces
    faces = []
    i, offset = 0, 0
    cc = smooth.faces 
    while offset < len(cc):
        nn = cc[offset]
        faces.append(cc[offset+1:offset+1+nn])
        offset += nn + 1
        i += 1

    # convert to triangles
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            triangles.append(face[:3])
            triangles.append(face[-3:])
        else:
            print(len(face))

    # create mesh
    mesh = [smooth.points, np.array(triangles)]
    
    return mesh


def rft_threshold(surface_file, q, outdir, alpha=5, n_iter=200, relax_factor=0.01, 
                  sigma=1., Z=50.0, output_thr=True, output_ec=True, output_resels=True):
    """
    Main function for deriving Random Field Theory thresholding [1,2] for a 
    function on a surface mesh. Core code based on [3].

    Parameters
    ----------
    surface_file : str
        Path to surface file in .asc format. Expects structure to be set out as
        follows (no header):
            
            <vertex_index_1> <x_1> <y_1> <z_1> <scalar_1>
            <vertex_index_2> <x_2> <y_2> <z_2> <scalar_2>
            ...
            
        Future implementations will receive inputs from other neuroimaging formats
        
    q : float
        Statistical alpha. This is (ideally) what the ec will be derived to,
        with equivalent Z threshold returning false positive rate equivalent
        to `q` (see [3])
    outdir : str
        Path to output directory for thresholded surface file. Also used to
        output vital statistics when `output_thr`, `output_ec`, and/or 
        `output_resel` are True
    alpha : int, optional
        Delaunay triangulation nearest neighbors connectivity. Increases the
        deformation of the convex hull (makes it larger). Higher or lower 
        values may be needed when degenerate triangles (holes) are present. 
        Default is 5
    n_iter : int, optional
        Number of surface smoothing iterations. Default is 200
    relax_factor : float, optional
        Triangulation relaxation factor. Defines how smooth local neighborhood 
        geometry is. Default is 0.01.
    sigma : float, optional
        Sigma for the Gaussian applied to the edge adjacency matrix of the
        surface. Enforces an approximate resel count. Default is 1.0
    Z : float, optional
        Maximum threshold to test at `sigma`. Smaller values make calculation
        of threshold quicker, but EC may not be found at statistical alpha `q`
    output_thr : bool, optional
        Specify whether to output threshold to text file in `outdir`. Default 
        is True
    output_ec : bool, optional
        Specify whether to output Euler characteristic function at Z and 
        `sigma` to a text file. Default is True
    output_resel : bool, optional
        Specify whether to output resel count at `sigma` to a text file. 
        Default is True

    References
    ----------
    [1] Worsley KJ, Evans AC, Marrett S, Neelin P. A three-dimensional 
    statistical analysis for CBF activation studies in human brain. J Cereb 
    Blood Flow Metab. 1992 Nov;12(6):900-18. doi: 10.1038/jcbfm.1992.127. 
    PMID: 1400644.
    
    [2] Worsley KJ, Marrett S, Neelin P, Vandal AC, Friston KJ, Evans AC. A 
    unified statistical approach for determining significant signals in images 
    of cerebral activation. Hum Brain Mapp. 1996;4(1):58-73. 
    doi: 10.1002/(SICI)1097-0193(1996)4:1<58::AID-HBM4>3.0.CO;2-O. 
    PMID: 20408186.
    
    [3] Brett, M. https://matthew-brett.github.io/teaching/random_fields.html
    
    """
    
    # get surface file name
    surface_filename = surface_file[:-4]
    
    # make surface
    v, t = make_surface(surface_file, alpha=alpha, n_iter=n_iter, 
                        relax_factor=relax_factor)
    
    # assumes last column is surface map
    vert_data = np.loadtxt(surface_file)[:, -1]
    
    # map vertex function to triangular function
    tria_data = map_vertex_func(vert_data, t)
    
    # make edge adjacency matrix
    adj = construct_adj_sym(tria_data, v, t)
    
    # find threshold
    z, ec, resels = euler_threshold(adj=adj, q=q, sigma=sigma, max_z=Z)
    
    # threshold values in data by z
    thresh = np.unique(np.argwhere(np.abs(vert_data) > z))
    thr_values = np.zeros_like(vert_data)
    thr_values[thresh] = vert_data[thresh]
    
    # write out thresholded values to asc file
    file_data = np.asarray([np.arange(1, v.shape[0]+1).astype(np.int8), v[:, 0], v[:, 1], v[:, 2], thr_values])
    file = f'{outdir}/{surface_filename}_z={z:0.4f}.asc'
    np.savetxt(file, file_data)
    
    m_print(f"Threshold at smoothing sigma={sigma} is Z={z} for statistical alpha q={q}.\nNumber of resels at sigma={sigma} is {resels}.\nThresholded surface saved to {file}")
    
    if output_thr:
        thr_file = f'{outdir}/{surface_filename}_z.txt'
        m_print(f"Output threshold saved to {thr_file}")
        np.savetxt(thr_file, z)
    
    if output_ec:
        ec_file = f'{outdir}/{surface_filename}_ec.txt'
        m_print(f"Output EC curve evaluated at Z=0 to Z={Z} saved to {ec_file}")
        np.savetxt(ec_file, ec)
        
    if output_resels:
        resels_file = f'{outdir}/{surface_filename}_resels.txt'
        m_print(f"Number of resels at sigma saved to {resels_file}")
        np.savetxt(resels_file, resels)


def main(raw_args=None):    
    parser = ArgumentParser(epilog="rft.py -- Implementation of RFT on surface adjacency matrices to find clusters at specified statistical alpha. Nikitas C. Koussis 2023 <nikitas.koussis@newcastle.edu.au>")
    parser.add_argument("surface_file", help="Surface file in .asc format, see help for specific structure")
    parser.add_argument("q", help="Statistical alpha for significance")
    parser.add_argument("outdir", help="Name of directory where outputs are to be stored")
    parser.add_argument("--alpha", default=5., help="Triangulation smoothing coefficient, default 5.0")
    parser.add_argument("--n_iter", default=200, help="Number of smoothing iterations when making the surface, default 200")
    parser.add_argument("--relax_factor", default=0.01, help="Relaxation factor of nearest neighbor smoothing, default 0.01")
    parser.add_argument("--sigma", default=1., help="Sigma for Gaussian smoothing kernel, measured in std dev, default=1.0")
    parser.add_argument("-Z", default=50.0, help="Default choice for maximum threshold to test, default=50.0")
    parser.add_argument("--output_thr", action='store_true', default=False, help="Specify whether to save thresholds to text file, useful for plotting EC")
    parser.add_argument("--output_ec", action='store_true', default=False, help="Specify whether to save EC curve based on different thresholds, useful for plotting EC curve with --output_thr")
    parser.add_argument("--output_resels", action='store_true', default=False, help="Specify whether to output number of resels to text file")
    
    #--------------------    Parsing the inputs from terminal:   -------------------
    args = parser.parse_args()
    surface_file         = args.surface_file
    q                    = float(args.q)
    outdir               = args.outdir
    alpha                = float(args.alpha)
    n_iter               = int(args.n_iter)
    relax_factor         = float(args.relax_factor)
    sigma                = float(args.sigma)
    Z                    = float(args.Z)
    output_thr           = args.output_thr
    output_ec            = args.output_ec
    output_resels        = args.output_resels
    #-------------------------------------------------------------------------------
   
    rft_threshold(surface_file, q, outdir, alpha, n_iter, relax_factor, sigma, Z, output_thr, output_ec, output_resels)
   

if __name__ == '__main__':
    
    # running via commandline
    main()
    
    
    