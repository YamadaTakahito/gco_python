# cython: experimental_cpp_class_def=True
import numpy as np
cimport numpy as np
from cpython cimport bool
import os

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        cppclass SmoothCostFunctor:
            int compute(int s1, int s2, int l1, int l2)

        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(int *)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        void setSmoothCostFunctor(SmoothCostFunctor* f)
        void setLabelOrder(bool RANDOM_LABEL_ORDER)
        int whatLabel(int node)
        void setVerbosity(int level)

cdef cppclass InpaintFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    int frame
    int height
    int width
    int img_size
    int n_labels
    int* imges
    int* knowns
    int* offsets

    __init__(int f_, int h_, int w_, int n_labels_, int* imges_, int* offsets_, int* knowns_):
        this.frame = f_
        this.height = h_
        this.width = w_
        this.img_size = h_ * w_
        this.n_labels = n_labels_
        this.imges = imges_
        this.offsets = offsets_
        this.knowns = knowns_

    int is_known(int f, int h, int w):
        idx = f * this.img_size  + h * this.width + w
        return this.knowns[idx] == 1
    
    int is_fit(int f, int h, int w):
        return f >= 0 and f < this.frame and h >= 0 and h < this.height and w >= 0 and w < this.width

    int is_valid(int f, int h, int w):
        return is_fit(f, h, w) and is_known(f, h, w)
    
    int compute_seam(int idx, int l1, int l2):
        cdef rest_idx = idx % (this.img_size)
        cdef int f = (idx - rest_idx) / this.img_size
        cdef int w = rest_idx % (this.width)
        cdef int h = (rest_idx - w) / this.width

        cdef int f1 = f + this.offsets[0 + 3 * l1]
        cdef int h1 = h + this.offsets[1 + 3 * l1]
        cdef int w1 = w + this.offsets[2 + 3 * l1]

        cdef int f2 = f + this.offsets[0 + 3 * l2]
        cdef int h2 = h + this.offsets[1 + 3 * l2]
        cdef int w2 = w + this.offsets[2 + 3 * l2]

        # print(this.fraame, this.height, this.width)
        # for destination pixels that are not known, bail with 0 energy
        # since single site infinity handles it
        if not is_valid(f1, h1, w1):
            return 100000
        if not is_valid(f2, h2, w2):
            return 100000
            
        cdef int c # color
        cdef int error
        cdef int i1
        cdef int i2
        cdef int idx1 = f1 * (this.height * this.width * 3) + h1 * (this.width * 3) + w1 * 3
        cdef int idx2 = f2 * (this.height * this.width * 3) + h2 * (this.width * 3) + w2 * 3
        c = 0

        # print(idx1, idx2)
        for c in range(3):
            i1 = this.imges[idx1 + c]
            i2 = this.imges[idx2 + c]
            i_diff = i1 - i2
            error += i_diff * i_diff
        return error
    
    int compute(int s1, int s2, int l1, int l2):
        # ||I(s1 + l1) - I(s1 + l2)||^2 + ||I(s2 + l1) - I(s2 + l2)||^2
        if(l1 == l2): return 0
        cdef int error1 = compute_seam(s1,l1,l2)
        cdef int error2 = compute_seam(s2,l1,l2)
        return error1 + error2


def cut_inpaint(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] offsets,
        np.ndarray[np.int32_t, ndim=4, mode='c'] imges,
        np.ndarray[np.int32_t, ndim=3, mode='c'] knowns,
        n_iter=5,
        algorithm='swap',
        randomizeOrder = False,
        verbosity = 0):
    """
    Apply multi-label graphcuts to grid graph using smoothing inpaint functor for
    pairwise costs

    Parameters
    ----------
    unary_cost: ndarray, int32, shape=(height, width, n_labels)
        Unary potentials
    offets: ndarray, int32, shape=(n_labels, 2)
        Offset for each label
    images: ndarray, int32, shape = (frame, height, width, 3)
        RGB video for calculating pairwise costs
    knowns: ndarray, int32, shape = (frame, height, width)
        Whether a pixel is in known or unknown region (1 = known, 0 unknown)
    n_iter: int, (default=5)
        Number of iterations
    algorithm: string, `expansion` or `swap`, default=expansion
        Whether to perform alpha-expansion or alpha-beta-swaps.
    randomizeOrder: boolean, default = False
        Whether to randomize min-cut order of swaps/expansions
    verbosity: int, (0 = none, 1 = medium, 2 = max)
        Control debug output from min-cut algorithm
    """

    if unary_cost.shape[2] != offsets.shape[0]:
        raise ValueError("unary_cost and offsets have incompatible shapes.\n"
            "unary_cost must be height x width x n_labels, offsets must be n_labels x 2.\n"
            "Got: unary_cost: (%d, %d, %d), pairwise_cost: (%d, %d)"
            %(unary_cost.shape[0], unary_cost.shape[1], unary_cost.shape[2],
                offsets.shape[0], offsets.shape[1]))
    if imges.shape[1] != unary_cost.shape[0] and imges.shape[2] != unary_cost.shape[1]:
        raise ValueError("unaray_cost shape must much image shape")
    if imges.shape[3] != 3:
        raise ValueError("Image must be RGB")
    if imges.shape[0] != knowns.shape[0] and imges.shape[1] != knowns.shape[1] and imges.shape[2] != knowns.shape[2]:
        raise ValueError("known shape must match image shape")

    cdef int frame = imges.shape[0]
    print(frame)
    cdef int height = imges.shape[1]
    cdef int width = imges.shape[2]
    cdef int n_labels = offsets.shape[0]

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(width, height, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostFunctor(<InpaintFunctor*>new InpaintFunctor(frame, height, width, n_labels, <int*>imges.data, <int*>offsets.data, <int*>knowns.data))
    if(randomizeOrder):
        print("Randomizing label order")
        gc.setLabelOrder(True)
    print("Verbosity {0}".format(verbosity))
    gc.setVerbosity(verbosity)
    if algorithm == 'swap':
        gc.swap(n_iter)
    elif algorithm == 'expansion':
        gc.expansion(n_iter)
    else:
        raise ValueError("algorithm should be either `swap` or `expansion`. Got: %s" % algorithm)

    cdef np.npy_intp result_shape[2]
    result_shape[0] = height
    result_shape[1] = width
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(width * height):
        result_ptr[i] = gc.whatLabel(i)
    return result
