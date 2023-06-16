#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "ICH.h"
#include "ExactMethodForDGP.h"


/*
    This program implements the fast marching algorithm in C
    based on the Improved Chen and Han's algorithm on the
    discrete geodesic problem for a vertex mesh.
    
    Can only input python numpy arrays.
*/

// initialize the actual function

static PyObject* geodesic_distmat(PyObject* self, PyObject* args) {
    PyArrayObject* vertices_array;
    PyArrayObject* faces_array;
    
    import_array();
    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &vertices_array, &PyArray_Type, &faces_array)) {
        return NULL;
    }
    
    int N = vertices_array->dimensions[0];
    
    npy_intp ndim = PyArray_NDIM(vertices_array)
    npy_intp fdim = PyArray_NDIM(faces_array)
    
    npy_intp dims[2] = {N, N};
    
    // initialize distance matrix
    PyArrayObject* distmat = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    
    if (ndim != 2 || fdim != 2)  {
        PyErr_SetString(PyExc_ValueError, "Inputs must be 2 dimensional");
        return NULL;
    }
    
    if (faces_array->dimensions[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Faces array must have shape (F, 3) where F is the number of faces");
        return NULL;    
    }

    double* vertices = (double *)PyArray_DATA(vertices_array);
    double* faces = (double *)PyArray_DATA(faces_array);
    double* distmat_data = (double*)PyArray_DATA(distmat);
    
    // Initialize the source vertex
    int index = 0;
    
    // Initialize the model
    ICHModel model;
    
    // Fill the vertices and faces
    
    
    // Preprocess
    model.LoadModel();
    model.Preprocess();
    
    // Run the algorithm
    CExactMethodforDGP* algorithm = new ImprovedCH(model, index);
    algorithm->Execute();
    
    // backtrace the shortest path of vertices
    
    // Now do the distance calculation for each vertex and fill the array distmat
    
    
    
    
    
    
    
    
    return PyArray_Return(distmat);


}

struct Window {
    // ... window properties ...
};

typedef struct {
    Vertex* vertices;
    Edge* edges;
    Face* faces;
    int num_vertices;
    int num_edges;
    int num_faces;
} Mesh;

typedef struct {
    Window* elements;
    int size;
    int capacity;
} PriorityQueue;

void update_distance(Vertex* v, double distance) {
    // ... update the distance estimate at vertex v ...
}

void add_child_window(PriorityQueue* Q, Window* w) {
    // ... add a child window to Q based on w ...
}

void compute_child_windows(PriorityQueue* Q, Window* w) {
    // ... compute child windows of w and add them to Q ...
}

void fast_marching(Mesh* mesh, Vertex* source) {
    // Initialize the priority queue Q
    PriorityQueue Q;
    // ... initialization code ...

    // Assign the source with distance 0
    source->distance = 0;

    // Create a pseudo-source window w for s, and put w into Q
    Window w;
    // ... initialization code for w ...
    // ... add w to Q ...

    // While Q is not empty and the level size doesn’t exceed the face number n
    while (Q.size > 0 && Q.size <= mesh->num_faces) {
        // Take out the head window w from Q
        Window w = pop(Q);

        // ... handle pseudo-source window ...

        // ... handle interval window ...

        // ... check if w can provide a shorter distance to the vertex v opposite to edge e ...

        // ... update distance estimates and add pseudo-source window to Q if necessary ...
    }
}
