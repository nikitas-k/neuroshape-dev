#define PY_SSIZE_T_CLEAN
#define NUMPY_INTERFACE
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include <stdio.h>

static PyMethodDef module_methods[] = {
    {"euler_threshold", compute_optimal_threshold, METH_VARARGS, "Compute optimal threshold using Euler characteristic."},
    {NULL, NULL, 0, NULL}
};

static int dfs(double* visited, double** similarity_matrix, double threshold, int vertex, int N) {
    visited[vertex] = 1.0;
    int euler_characteristic = 1;

    for (int i = 0; i < N; i++) {
        if (visited[i] == 0.0 && similarity_matrix[vertex][i] > threshold) {
            euler_characteristic += dfs(visited, similarity_matrix, threshold, i, N);
        }
    }

    return euler_characteristic;
}

static int compute_euler_characteristic(double* visited, double** similarity_matrix, double threshold, int N) {
    int euler_characteristic = 0;

    for (int i = 0; i < N; i++) {
        if (visited[i] == 0.0) {
            euler_characteristic += dfs(visited, similarity_matrix, threshold, i, N);
        }
    }

    return euler_characteristic;
}

static PyObject* euler_threshold(PyObject* self, PyObject* args) {
    PyArrayObject* similarity_matrix_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &similarity_matrix_array)) {
        return NULL;
    }

    int N = PyArray_DIM(similarity_matrix_array, 0);
    double** similarity_matrix = (double**)PyArray_DATA(similarity_matrix_array);

    double optimal_threshold = 0.0;
    double step_size = 1.0 / 256;
    int max_euler_characteristic = 0;

    double* visited = (double*)calloc(N, sizeof(double));

    for (int i = 0; i <= 256; i++) {
        double threshold = i * step_size;

        int euler_characteristic = compute_euler_characteristic(visited, similarity_matrix, threshold, N);
        if (euler_characteristic > max_euler_characteristic) {
            max_euler_characteristic = euler_characteristic;
            optimal_threshold = threshold;
        }
    }

    free(visited);

    return Py_BuildValue("d", optimal_threshold);
}

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "euler_threshold",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_euler(void) {
    PyObject* module;
    module = PyModule_Create(&module_def);
    if (module == NULL)
        return NULL;
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}