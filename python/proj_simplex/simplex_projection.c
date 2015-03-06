/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

/* averages y[start...end] (exclusive) */
static double block_avg(double *y, int arr_start, int start, int end) {
  if (start < 0 || end < 0) {
    /* HACK (raises error) */
    printf("HACK\n");
    return 1/0;
  } else if(start >= end) {
    return y[arr_start+start];
  }
  int n = end - start;
  double avg = 0;
  for (; start < end; ++start) {
    avg += y[arr_start + start];
  }
  return avg / n;
}

void pop(int *arr, int n, int idx) {
  for (; idx < n - 1; ++idx) {
    arr[idx] = arr[idx + 1];
  }
  /* Should probably be filled with NaN's. Hopefully this is just
   * a hack until a linked-list implementation is made */
  arr[n - 1] = -1;
}

static void pav_algorithm(double *y, double *x, int n, double l, double u, int start, int end) {
  double avg;
  int used_n, i, ell, ind;
  int *j;

  if (n == 1) {
    avg = y[start];
    if (avg < l) {
      avg = l;
    } else if (avg > u) {
      avg = u;
    }
    x[start] = avg;
  } else if (n == 2) {
    if (y[start] > y[start + 1]) {
      avg = block_avg(y, start, 0, 2);
      if (avg < l) {
        avg = l;
      } else if (avg > u) {
        avg = u;
      }
      x[start] = avg;
      x[start + 1] = avg;
    } else {
      avg = y[start];
      if (avg < l) {
        avg = l;
      } else if (avg > u) {
        avg = u;
      }
      x[start] = avg;
      avg = y[start + 1];
      if (avg < l) {
        avg = l;
      } else if (avg > u) {
        avg = u;
      }
      x[start + 1] = avg;
    }
  } else if (n > 2) {
    /* TODO: replace with linked list */
    used_n = n+1;
    j = (int*)malloc(sizeof(int)*used_n);
    for (i = 0; i < used_n; ++i) {
      j[i] = i;
    }
    ind = start;
    while (ind < used_n-2) {
      if (block_avg(y, start, j[ind + 1], j[ind + 2]) < block_avg(y, start, j[ind], j[ind + 1])) {
        pop(j, used_n, ind+1);
        --used_n;
        while ((ind > 0) && (ind < used_n-1) &&
            (block_avg(y, start, j[ind - 1], j[ind]) > block_avg(y, start, j[ind], j[ind + 1]))) {
          if (block_avg(y, start, j[ind], j[ind + 1])
              <= block_avg(y, start, j[ind - 1], j[ind])) {
            pop(j, used_n, ind);
            --used_n;
            ind -= 1;
          }
        }
      } else {
        ind += 1;
      }
    }
    for (i = 0; i < used_n - 1; ++i) {
      ell = j[i+1];
      if (ell > n) {
        avg = block_avg(y, start, j[i], n);
      } else {
        avg = block_avg(y, start, j[i], ell);
      }
      if (avg < l) {
        avg = l;
      } else if (avg > u) {
        avg = u;
      }
      for (ell = j[i]; (ell < n) && (ell < j[i + 1]); ++ell) {
        x[ell + start] = avg;
      }
    }
    free(j);
    j = NULL;
  }
}

static PyObject* pav_projection(PyObject* self, PyObject* args)
{

    PyArrayObject *y_np;
    double *y, *x;
    double u, l, avg;
    int n;
    PyObject      *out_array;

    /*  parse single numpy array argument */
    /* Arguments are y (np array), l (double), u (double) */
    if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &y_np, &l, &u))
        return NULL;

    y_np = PyArray_Cast(y_np, NPY_DOUBLE);

    /*  construct the output array, like the y array */
    out_array = PyArray_NewLikeArray(y_np, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    n = PyArray_DIM(y_np, 0);
    y = (double *)PyArray_DATA(y_np);
    x = (double *)PyArray_DATA(out_array);

    pav_algorithm(y, x, n, l, u, 0, n);

    Py_DECREF(y_np);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

static PyObject* simplex_projection(PyObject* self, PyObject* args)
{

    PyArrayObject *y_np, *N;
    PyObject      *out_array;
    double *y, *x;
    int *blocks;
    int blk_size, n, blk_start, n_blocks, i;

    blk_start = 0;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &N, &PyArray_Type, &y_np))
        return NULL;

    y_np = PyArray_Cast(y_np, NPY_DOUBLE);
    N = PyArray_Cast(N, NPY_INT32);

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(y_np, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    n = PyArray_DIM(y_np, 0);
    y = (double *)PyArray_DATA(y_np);
    x = (double *)PyArray_DATA(out_array);

    n_blocks = PyArray_DIM(N, 0);
    blocks = (int *)PyArray_DATA(N);

    for (i = 0; i < n_blocks; ++i) {
        blk_size = blocks[i];
        pav_algorithm(y, x, blk_size, 0., 1., blk_start, blk_start + blk_size);
        blk_start += blk_size;
    }

    /*  clean up and return the result */
    Py_DECREF(N);
    Py_DECREF(y_np);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

/*  define functions in module */
static PyMethodDef CosMethods[] =
{
     {"simplex_projection", simplex_projection, METH_VARARGS,
         "evaluate the simplex_projection on a numpy array"},
     {"pav_projection", pav_projection, METH_VARARGS,
         "evaluate the pav_projection on a numpy array"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initsimplex_projection(void)
{
     (void) Py_InitModule("simplex_projection", CosMethods);
     /* IMPORTANT: this must be called */
     import_array();
}
