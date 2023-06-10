#define PY_SSIZE_T_CLEAN
#define NUMPY_INTERFACE
#include <Python.h>
#include <numpy/arrayobject.h>
#include "BaseModel.h"
#include "Point3D.h"
#include "geo.h"

// Wrapper for calculate_all_geodesic_distances function
static PyObject* geodesics_improvedch(PyObject* self, PyObject* args) {
    PyArrayObject* vertex_array;
    PyArrayObject* faces_array;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &vertex_array, &PyArray_Type, &faces_array)) {
        return NULL;
    }

    int N = vertex_array->dimensions[0];
    
    // Convert numpy arrays to C++ vectors
    std::vector<CPoint3D> vertices((CPoint3D*)PyArray_DATA(vertex_array), (CPoint3D*)PyArray_DATA(vertex_array) + PyArray_SIZE(vertex_array));
    std::vector<CFace> faces((CFace*)PyArray_DATA(faces_array), (CFace*)PyArray_DATA(faces_array) + PyArray_SIZE(faces_array));

    // Call the C++ function
    std::vector<double> distances = calculate_all_geodesic_distances(std::vector<CPoint3D> & vertices, std::vector<CFace> & faces);

    // Convert the output to a numpy array
    npy_intp dims[2] = { N, N };
    
    // Create an array to hold the float values
    float* float_distances = malloc(num_vertices * num_vertices * sizeof(float));
    
    // Convert each double value to float
    for (int i = 0; i < N * N; i++) {
        float_distances[i] = (float)distances[i];
    }
    
    // Create the numpy array using the float data
    PyObject* distmat = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, float_distances);

    // The C++ std::vector will be cleaned up when it goes out of scope, so we need to let numpy know it has to make its own copy of the data
    PyArray_ENABLEFLAGS((PyArrayObject*)distmat, NPY_ARRAY_OWNDATA);

    // Remember to free the allocated memory once you're done
    free(float_distances);
    
    return distmat;
}

// Module method definitions
static PyMethodDef GeodesicsMethods[] = {
    {"geodesics_improvedch", geodesics_improvedch, METH_VARARGS, "Calculates all geodesic distances using improved Chen and Han's algorithm"},
    {NULL, NULL, 0, NULL}  // Sentinel value ending the array
};

// Module structure
static struct PyModuleDef geodesicsmodule = {
    PyModuleDef_HEAD_INIT,
    "geodesics",  // Name of the module
    "A module that wraps improved Chen and Han's algorithm for geodesic distance calculation",  // Module documentation
    -1,  // Per-interpreter state of the module, typically -1
    GeodesicsMethods  // Structure that defines the methods of the module
};

// Module initialization
// The initialization function must be named PyInit_<name>
PyMODINIT_FUNC
PyInit_geodesics(void)
{
    PyObject* module;

    // Initialize numpy functionality
    import_array();

    module = PyModule_Create(&geodesicsmodule);
    if (module == NULL)
        return NULL;

    return module;
}