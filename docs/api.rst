.. _api_ref:

.. currentmodule:: neuroshape

Reference API
=============

.. contents:: **List of modules**
   :local:

.. _ref_shape:

:mod:`neuroshape.shape` - Mesh base
-----------------------------------
.. automodule:: neuroshape.shape
   :no-members:
   :no-inherited-members:

.. currentmodule:: neuroshape.shape

.. autosummary::
   :template: class.rst
   :toctree: generated/

   neuroshape.shape.read_surface
   neuroshape.shape.write_surface
   neuroshape.shape.read_data
   neuroshape.shape.write_data
   neuroshape.shape.read_mask
   neuroshape.shape.write_mask
   neuroshape.shape.read_emodes
   neuroshape.shape.write_emodes
   neuroshape.shape.read_evals
   neuroshape.shape.get_verts
   neuroshape.shape.get_faces
   neuroshape.shape.get_edges
   neuroshape.shape.get_polyshape
   neuroshape.shape.get_closed
   neuroshape.shape.sort_verts
   neuroshape.shape.find_vertex_correspondence
   neuroshape.shape.get_vertex_neighbors
   neuroshape.shape.has_free_vertices
   neuroshape.shape.remove_free_vertices
   neuroshape.shape.vertex_degrees
   neuroshape.shape.vertex_areas
   neuroshape.shape.vertex_normals
   neuroshape.shape.shortest_path_length
   neuroshape.shape.geodesic_distance
   neuroshape.shape.geodesic_knn
   neuroshape.shape.apply_mask
   neuroshape.shape.parcellate_data
   neuroshape.shape.threshold_surface
   neuroshape.shape.threshold_minimum
   neuroshape.shape.combine_surfaces
   neuroshape.shape.split_surface
   neuroshape.shape.normalize
   neuroshape.shape.unit_normalize
   neuroshape.shape.refine
   neuroshape.shape.orient
   neuroshape.shape.smooth
   neuroshape.shape.decimate
   neuroshape.shape.decimate_pro
   neuroshape.shape.optimize_recon
   neuroshape.shape.reconstruct_data
   neuroshape.shape.to_tetrahedral
   neuroshape.shape.to_triangular
   neuroshape.shape.isoriented
   neuroshape.shape.average_edge_length
   neuroshape.shape.area
   neuroshape.shape.volume
   neuroshape.shape.boundary_loops
   neuroshape.shape.centroid
   neuroshape.shape.curvature
   neuroshape.shape.modularity
   neuroshape.shape.geodesic_distmat
   neuroshape.shape.euclidean_distmat
   neuroshape.shape.tria_normals
   neuroshape.shape.tria_qualities
   neuroshape.shape.tria_areas
   neuroshape.shape.is_manifold
   neuroshape.shape.euler
   neuroshape.shape.compute_eigenmodes
