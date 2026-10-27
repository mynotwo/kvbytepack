#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

/* Read/store byte lanes explicitly so the kernel is endian-independent.
 * Compilers on little-endian targets fold these into unaligned word accesses. */
static uint64_t load_le(const unsigned char *p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i);
    return x;
}
static void store_le(unsigned char *p, uint64_t x) {
    for (int i = 0; i < 8; ++i) p[i] = (unsigned char)(x >> (8 * i));
}
static void exchange(uint64_t *a, uint64_t *b, int shift, uint64_t mask) {
    uint64_t t = ((*a >> shift) ^ *b) & mask;
    *a ^= t << shift;
    *b ^= t;
}
/* Transpose an 8x8 matrix of bytes, using three butterfly stages. */
static void transpose8(uint64_t x[8]) {
    for (int i = 0; i < 4; ++i)
        exchange(&x[i], &x[i + 4], 32, UINT64_C(0x00000000ffffffff));
    for (int b = 0; b < 8; b += 4)
        for (int i = b; i < b + 2; ++i)
            exchange(&x[i], &x[i + 2], 16, UINT64_C(0x0000ffff0000ffff));
    for (int i = 0; i < 8; i += 2)
        exchange(&x[i], &x[i + 1], 8, UINT64_C(0x00ff00ff00ff00ff));
}
static uint64_t spread4(uint32_t value) {
    uint64_t x = value;
    x = (x | (x << 16)) & UINT64_C(0x0000ffff0000ffff);
    return (x | (x << 8)) & UINT64_C(0x00ff00ff00ff00ff);
}
/* Two immutable byte planes -> one private output; no retained scratch. */
static PyObject *merge_bf16(PyObject *self, PyObject *args) {
    PyObject *low, *high, *result;
    Py_ssize_t tokens, channels, count;
    const unsigned char *a, *b;
    unsigned char *out;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOnn:merge_bf16", &low, &high, &tokens, &channels))
        return NULL;
    if (!PyBytes_CheckExact(low) || !PyBytes_CheckExact(high)) {
        PyErr_SetString(PyExc_TypeError, "planes must be immutable bytes");
        return NULL;
    }
    if (tokens < 0 || channels < 0) {
        PyErr_SetString(PyExc_ValueError, "dimensions must be nonnegative");
        return NULL;
    }
    if (channels && tokens > PY_SSIZE_T_MAX / channels / 2) {
        PyErr_SetString(PyExc_OverflowError, "decoded size exceeds addressable bytes");
        return NULL;
    }
    count = tokens * channels;
    if (PyBytes_GET_SIZE(low) != count || PyBytes_GET_SIZE(high) != count) {
        PyErr_SetString(PyExc_ValueError, "plane size does not match geometry");
        return NULL;
    }
    result = PyBytes_FromStringAndSize(NULL, count * 2);
    if (result == NULL || count == 0)
        return result;
    a = (const unsigned char *)PyBytes_AS_STRING(low);
    b = (const unsigned char *)PyBytes_AS_STRING(high);
    out = (unsigned char *)PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    Py_ssize_t cf = channels - channels % 8, tf = tokens - tokens % 8;
    for (Py_ssize_t cb = 0; cb < cf; cb += 8) {
        for (Py_ssize_t tb = 0; tb < tf; tb += 8) {
            uint64_t lo[8], hi[8];
            for (int c = 0; c < 8; ++c) {
                Py_ssize_t src = (cb + c) * tokens + tb;
                lo[c] = load_le(a + src);
                hi[c] = load_le(b + src);
            }
            transpose8(lo);
            transpose8(hi);
            for (int t = 0; t < 8; ++t) {
                unsigned char *dst = out + ((tb + t) * channels + cb) * 2;
                store_le(dst, spread4((uint32_t)lo[t]) | (spread4((uint32_t)hi[t]) << 8));
                store_le(dst + 8, spread4((uint32_t)(lo[t] >> 32)) |
                         (spread4((uint32_t)(hi[t] >> 32)) << 8));
            }
        }
    }
    /* Partial tiles preserve arbitrary and tiny tensor geometry. */
    for (Py_ssize_t t = tf; t < tokens; ++t) {
        for (Py_ssize_t c = 0; c < channels; ++c) {
            Py_ssize_t dst = (t * channels + c) * 2, src = c * tokens + t;
            out[dst] = a[src]; out[dst + 1] = b[src];
        }
    }
    for (Py_ssize_t c = cf; c < channels; ++c) {
        for (Py_ssize_t t = 0; t < tf; ++t) {
            Py_ssize_t dst = (t * channels + c) * 2, src = c * tokens + t;
            out[dst] = a[src]; out[dst + 1] = b[src];
        }
    }
    Py_END_ALLOW_THREADS
    return result;
}

static PyMethodDef methods[] = {
    {"merge_bf16", merge_bf16, METH_VARARGS, "Merge two byte planes into token-major bytes."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_decode", "Optional byte-layout decode kernel.", 0, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__decode(void) {
    return PyModule_Create(&module);
}
