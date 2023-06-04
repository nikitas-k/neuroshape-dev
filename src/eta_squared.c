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

/* declare main function */
static PyObject* eta_squared(PyObject* self, PyObject* args);

/*
 * This tells Python what methods this module has
 *
 */
static PyMethodDef EtaMethods[] = {
    {"eta_squared",
        eta_squared,
        METH_VARARGS, "Compute eta-squared coefficient row-wise of a 2-dimensional array."},
    {NULL, NULL, 0, NULL}
};

/*
 * This actually defines the eta squared function for
 * input args from Python.
 */

static PyObject* eta_squared(PyObject* self, PyObject* args)
{
    PyArrayObject *arr, *oarr;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) return NULL;
    
//     arr = (PyArrayObject *)
//         PyArray_ContiguousFromObject(arg1, NPY_DOUBLE, 2, 2);
//    if (arr == NULL) return NULL;
    
    int num_rows, num_cols;
    num_rows = arr -> dimensions[0];
    num_cols = arr -> dimensions[1];
    npy_intp dims[2] = {num_rows,num_rows};
    int nd = 2;
    
    oarr = (PyArrayObject *) PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    import_array();
        
//    if (oarr == NULL) goto fail;
    
    /* code that makes use of arguments */
    double *iptr=NULL, *jptr=NULL, *kptr=NULL, *iiptr=NULL, *outptr=NULL;
    
    npy_intp i,j,k;
    
    iptr = (double *)PyArray_DATA(arr);
    iiptr = (double *)PyArray_DATA(arr);
    jptr = (double *)PyArray_DATA(arr);
    kptr = (double *)PyArray_DATA(arr);
    outptr = (double *)PyArray_DATA(oarr);
    
    if (iptr == NULL) return NULL;
    if (jptr == NULL) return NULL;
    if (kptr == NULL) return NULL;
    if (outptr == NULL) return NULL;
    
    double mu_bar;
    double mu_1 = 0;
    double vali, valk;
    int count = 0;
    
    /* Get mu_bar by taking mu over every pair */
    for (i = 0; i < num_rows; i++){
        for (k = 0; k< num_rows; k++){
            for (j = 0; j < num_cols; j++){
                iptr = (arr->data + i*arr->strides[0] +
                    j*arr->strides[1]);
                vali = iptr[0];
                iiptr = (arr->data + k*arr->strides[0] +
                    j*arr->strides[1]);
                valk = iiptr[0];
                
                mu_1 += ((vali + valk) / 2);
                count++;
            }
        }
    }
    mu_bar = (mu_1 / count);
    
    /* Main loop */
    for (i = 0; i < num_rows; i++){    
        for (k = 0; k < num_rows; k++){
            double num = 0;
            double denom = 0;
            for (j = 0; j < num_cols; j++){
                jptr = (arr->data + i*arr->strides[0] +
                    j*arr->strides[1]);
                vali = jptr[0];
                kptr = (arr->data + k*arr->strides[0] +
                    j*arr->strides[1]);
                valk = kptr[0];
                
                double mu = ((vali + valk) / 2);
                double powera = pow((vali - mu), (double)2);
                double powerb = pow((valk - mu), (double)2);
                
                double powerax = pow((vali - mu_bar), (double)2);
                double powerbx = pow((valk - mu_bar), (double)2);
                
                num += (powera + powerb);
                denom += (powerax + powerbx);
            }
            double eta = 1 - (num / denom);
            
            outptr = (oarr->data + i*oarr->strides[0] +
                        k*oarr->strides[1]);
            *outptr = eta;
        }
    }
    
    Py_DECREF(outptr);
    return PyArray_Return(oarr);
    
    fail:
        Py_XDECREF(arr);
#if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy(oarr);
#endif
        Py_XDECREF(oarr);
        return NULL;
}

/* This initiates the module using the above definitions. */
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "neuroshape.eta", "Eta-squared 2-dimensional array calculation implementation in C",
    -1,
    EtaMethods
};

PyMODINIT_FUNC PyInit_eta(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if(module==NULL) return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred()) return NULL;
    return module;
}