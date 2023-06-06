#define PY_SSIZE_T_CLEAN
#define NUMPY_INTERFACE
#include <stdlib.h>
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include <stdio.h>

/* Function to perform LU decomposition */
int luDecomposition(double* A, int n, int* pivot)
{
    for (int i = 0; i < n; i++) {
        pivot[i] = i;
    }

    for (int k = 0; k < n; k++) {
        double max = 0.0;
        int maxIndex = k;

        for (int i = k; i < n; i++) {
            double absValue = fabs(A[i * n + k]);
            if (absValue > max) {
                max = absValue;
                maxIndex = i;
            }
        }

        if (max == 0.0) {
            return -1; // Matrix is singular
        }

        if (maxIndex != k) {
            // Swap rows
            for (int j = 0; j < n; j++) {
                double temp = A[k * n + j];
                A[k * n + j] = A[maxIndex * n + j];
                A[maxIndex * n + j] = temp;
            }

            // Swap pivot entries
            int temp = pivot[k];
            pivot[k] = pivot[maxIndex];
            pivot[maxIndex] = temp;
        }

        double pivotValue = A[k * n + k];
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= pivotValue;
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }

    return 0;
}

/* Function to solve a linear system using LU decomposition */
void luSolve(double* LU, int* pivot, double* b, int n, double* x)
{
    for (int i = 0; i < n; i++) {
        x[i] = b[pivot[i]];
        for (int j = 0; j < i; j++) {
            x[i] -= LU[i * n + j] * x[j];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            x[i] -= LU[i * n + j] * x[j];
        }
        x[i] /= LU[i * n + i];
    }
}

/* Function to perform glmfit */
static PyObject* glmfit(PyObject* self, PyObject* args)
{
    PyObject* x_obj;
    PyObject* y_obj;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj)) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments");
        return NULL;
    }

    // Convert NumPy arrays to C arrays
    PyArrayObject* x_array = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL || y_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert input arrays to C arrays");
        return NULL;
    }

    // Get dimensions and pointers to the data
    int n = (int)PyArray_DIM(x_array, 0);
    int p = (int)PyArray_DIM(x_array, 1);
    double* x = (double*)PyArray_DATA(x_array);
    double* y = (double*)PyArray_DATA(y_array);

    // Allocate memory for matrices
    double* X = (double*)malloc(n * p * sizeof(double));
    double* Xt = (double*)malloc(p * n * sizeof(double));
    double* XtX = (double*)malloc(p * p * sizeof(double));
    double* XtX_inv = (double*)malloc(p * p * sizeof(double));
    double* XtX_inv_Xt = (double*)malloc(p * n * sizeof(double));
    int* pivot = (int*)malloc(p * sizeof(int));

    // Fill X matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            X[i * p + j] = x[i * p + j];
        }
    }

    // Transpose X matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Xt[j * n + i] = X[i * p + j];
        }
    }

    // Compute XtX matrix
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += Xt[i * n + k] * X[k * p + j];
            }
            XtX[i * p + j] = sum;
        }
    }

    // Perform LU decomposition of XtX matrix
    if (luDecomposition(XtX, p, pivot) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "Matrix is singular. Unable to compute inverse.");
        return NULL;
    }

    // Compute inverse of XtX matrix using LU decomposition
    for (int i = 0; i < p; i++) {
        double b[p];
        for (int j = 0; j < p; j++) {
            b[j] = (i == j) ? 1.0 : 0.0;
        }

        luSolve(XtX, pivot, b, p, XtX_inv + i * p);
    }

    // Compute XtX_inv_Xt matrix
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < p; k++) {
                sum += XtX_inv[i * p + k] * Xt[k * n + j];
            }
            XtX_inv_Xt[i * n + j] = sum;
        }
    }

    // Compute beta coefficients
    double* beta = (double*)malloc(p * sizeof(double));
    for (int i = 0; i < p; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += XtX_inv_Xt[i * n + j] * y[j];
        }
        beta[i] = sum;
    }

    // Create a NumPy array to store the beta coefficients
    npy_intp dims[1] = { p };
    PyObject* beta_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, beta);

    // Clean up allocated memory
    free(X);
    free(Xt);
    free(XtX);
    free(XtX_inv);
    free(XtX_inv_Xt);
    free(pivot);

    // Return the beta coefficients as a NumPy array
    return beta_array;
}

/* Module method table */
static PyMethodDef GLMMethods[] = {
    {"glmfit", (PyCFunction)glmfit, METH_VARARGS, "Compute beta coefficients using glmfit."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "glmfit",
    NULL,
    -1,
    GLMMethods
};

PyMODINIT_FUNC PyInit_glmfit(void)
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

