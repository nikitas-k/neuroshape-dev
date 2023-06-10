#include <Python.h>
#include <numpy/arrayobject.h>
#include ".h"

// Wrapper for calculate_all_geodesic_distances function
static PyObject* geo_lib_calculate_all_geodesic_distances(PyObject* self, PyObject* args) {
    char* filename;
    int num_vertices;
    int num_faces;

    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "sii", &filename, &num_vertices, &num_faces)) {
        return NULL;
    }

    // Call the C++ function
    double* distances = calculate_all_geodesic_distances(filename, num_vertices, num_faces);

    // Convert the output to a numpy array
    npy_intp dims[2] = { num_vertices, num_vertices };
    PyObject* result = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, distances);

    return result;
}

// Method list
static PyMethodDef GeoLibMethods[] = {
    {"calculate_all_geodesic_distances", geo_lib_calculate_all_geodesic_distances, METH_VARARGS,
     "Calculate all geodesic distances in a model."},
    {NULL, NULL, 0, NULL}
};

// Module structure
static struct PyModuleDef geo_lib_module = {
    PyModuleDef_HEAD_INIT,
    "geo_lib",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    GeoLibMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_geo_lib(void) {
    PyObject* m;

    m = PyModule_Create(&geo_lib_module);
    if (m == NULL)
        return NULL;

    // IMPORTANT: this must be called to initialize numpy array API
    import_array();

    return m;
}