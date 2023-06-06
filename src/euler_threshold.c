#define PY_SSIZE_T_CLEAN
#define NUMPY_INTERFACE
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include <stdio.h>

static double expected_ec(double z, int num_resels) {
    double ec = num_resels * (4.0 * log(2.0)) * pow((2.0 * M_PI), (-3.0 / 2.0)) * z * exp(-0.5 * pow(z, 2.0));
    return ec;
}

static PyObject* euler_threshold(PyObject* self, PyObject* args) {
    PyArrayObject* arr;
    double num_resels;

    /* Parse the input arguments */
    if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &arr, &num_resels)) {
        return NULL;
    }

    /* Check if the input array is 2D and of type double */
    if (PyArray_NDIM(arr) != 2 || PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input array must be 2D and of type double");
        return NULL;
    }

    /* Get the dimensions and strides of the input array */
    npy_intp rows = PyArray_DIM(arr, 0);
    npy_intp cols = PyArray_DIM(arr, 1);
    npy_intp row_stride = PyArray_STRIDE(arr, 0) / sizeof(double);
    npy_intp col_stride = PyArray_STRIDE(arr, 1) / sizeof(double);
    double threshold;

    /* Compute the optimal threshold for the entire input array */
    double* input_data = PyArray_DATA(arr);
    double max_ec = -INFINITY;
    for (npy_intp i = 0; i < rows; i++) {
        for (npy_intp j = 0; j < cols; j++) {
            double value = input_data[i * row_stride + j * col_stride];
            double ec = expected_ec(value, num_resels);
            if (ec > max_ec) {
                max_ec = ec;
                threshold = value;
            }
        }
    }

    /* Create a Python float object to hold the result */
    PyObject* result = PyFloat_FromDouble(threshold);
    if (result == NULL) {
        return NULL;
    }

    return result;

}

static PyMethodDef EulerMethods[] = {
    {"euler_threshold", euler_threshold, METH_VARARGS, "Compute optimal threshold using Euler characteristic. Parameters: array : 2d np.ndarray of float32 or float64. int : number of resels. Returns: optimal_threshold : double"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "neuroshape.euler",
    "Backend C statistical analyses.",
    -1,
    EulerMethods
};

PyMODINIT_FUNC PyInit_euler(void)
{
    PyObject *m;
    
    import_array();
    import_umath();
    
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    
    return m;
}