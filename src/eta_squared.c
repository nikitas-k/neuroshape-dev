#define PY_SSIZE_T_CLEAN
#define NUMPY_INTERFACE
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>
#include <math.h>
#include <stdio.h>

/*
 * eta_squared.c
 * This program is a non-numpy Python extension to compute
 * the eta-squared coefficient quicker, where
 * 
 *
 *                 sum ( ( a_i - mu ) ^ 2 + ( b_i - mu ) ^ 2 )
 * eta^2 = 1 -  -------------------------------------------------
 *              sum ( ( a_i - mu_bar ) ^ 2 + ( b_i - mu_bar ) ^ 2)
 *
 * where a is the first set of observations and b is the second set
 * of observations. mu is the average of a_i and b_i (i.e., 
 * ( a_i + b_i ) / 2 ), and mu_bar is the average of all observations.
 *
 * For this program to work, numpy.vectorize must be called in
 * in python to generate a numpy-friendly function. See 
 * neuroshape/examples/compute_eta_squared.py for an example
 * 
 */

/*
 * This actually defines the eta squared function for
 * input args from Python.
 */


static PyObject* eta_squared(PyObject* self, PyObject* args)
{
    PyArrayObject* arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr))
        return NULL;
    
    int N = arr->dimensions[0];
    int p = arr->dimensions[1];
    
    /* check number of dims and whether data is in correct format */
    if ( N < p ) {
        printf('ERROR: Number of observations must exceed number of features (is your data transposed?)');
        return NULL;    
    }
    npy_intp nd = PyArray_NDIM(arr);
    if ( nd != 2 ) {
        printf('ERROR: Input must be a 2-dimensional matrix');
        return NULL;    
    }
    
    npy_intp dims[2] = {N, N};

    PyArrayObject* oarr = (PyArrayObject*)PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    if (oarr == NULL)
        return NULL;

    double* z = PyArray_DATA(arr);
    double* eta = PyArray_DATA(oarr);

    double* mu = (double*)malloc(N * p * sizeof(double));
    double* mu_bar = (double*)malloc(N * sizeof(double));
    
    npy_intp row_stride = PyArray_STRIDE(arr, 0) / sizeof(double);
    npy_intp col_stride = PyArray_STRIDE(arr, 1) / sizeof(double);
    npy_intp outrow_stride = PyArray_STRIDE(oarr, 0) / sizeof(double);
    npy_intp outcol_stride = PyArray_STRIDE(oarr, 1) / sizeof(double);
    
    for (int i = 0; i < N; i++) {
        // Calculate mu
        for (int k = 0; k < p; k++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i)
                    sum += z[j * row_stride + k * col_stride];
            }
            mu[k] = (z[i * row_stride + k * col_stride] + sum) / N;
        }


        // Calculate mu_bar
        for (int k = 0; k < p; k++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += z[j * p + k];
            }
            mu_bar[k] = sum / N;
        }

        // Calculate eta_squared
        for (int j = 0; j < N; j++) {
            double num = 0.0;
            double denom = 0.0;
            for (int k = 0; k < p; k++) {
                double diff = z[i * row_stride + k * col_stride] - mu[k];
                num += diff * diff;
                double diff_bar = z[j * row_stride + k * col_stride] - mu_bar[k];
                denom += diff_bar * diff_bar;
            }
            eta[j * outrow_stride + i * outcol_stride] = 1.0 - (num / denom);
        }
    }

    free(mu);
    free(mu_bar);

    return PyArray_Return(oarr);
}
//     // Calculate eta-squared coefficients
//     for (i = 0; i < num_rows; i++) {
//         for (k = 0; k < num_rows; k++) {
//             double num = 0.0;
//             double denom = 0.0;
//             for (j = 0; j < num_cols; j++) {
//                 double vali = data_ptr[i * row_stride + j * col_stride];
//                 double valk = data_ptr[k * row_stride + j * col_stride];
//                 double mu = (vali + valk) / 2.0;
//                 double diff_a = vali - mu;
//                 double diff_b = valk - mu;
//                 double powera = diff_a * diff_a;
//                 double powerb = diff_b * diff_b;
//                 double diff_bar_a = vali - mu_bar;
//                 double diff_bar_b = valk - mu_bar;
//                 double powerax = diff_bar_a * diff_bar_a;
//                 double powerbx = diff_bar_b * diff_bar_b;
//                 num += powera + powerb;
//                 denom += powerax + powerbx;
//             }
//             if ( denom == 0.0 ){
//                 double eta = 0.0;
//                 out_ptr[i*num_rows + k] = eta;
//             } else {
//                 double eta = 1.0 - (num / denom);
//                 out_ptr[i*num_rows + k] = eta;
//             }
//             
//         }
//     }


static PyMethodDef EtaMethods[] = {
    {"eta_squared",
        eta_squared,
        METH_VARARGS, "Compute eta-squared coefficient row-wise of a 2-dimensional array."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "neuroshape.eta",
    "Eta-squared 2-dimensional array calculation implementation in C",
    -1,
    EtaMethods
};

PyMODINIT_FUNC PyInit_eta(void)
{
    PyObject* module;
    module = PyModule_Create(&cModPyDem);
    if (module == NULL)
        return NULL;
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}