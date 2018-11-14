# cython: experimental_cpp_class_def=True
import numpy as np
cimport numpy as np
from cpython cimport bool

np.import_array()

cdef extern from "GCoptimization.h":
    cdef cppclass GCoptimizationGridGraph:
        cppclass SmoothCostFunctor:
            int compute(int s1, int s2, int l1, int l2)

        GCoptimizationGridGraph(int width, int height, int n_labels)
        void setDataCost(int *)
        void setSmoothCost(int *)
        void expansion(int n_iterations)
        void swap(int n_iterations)
        void setSmoothCostVH(int* pairwise, int* V, int* H)
        void setSmoothCostFunctor(SmoothCostFunctor* f)
        void setLabelOrder(bool RANDOM_LABEL_ORDER)
        int whatLabel(int node)
        void setVerbosity(int level)

cdef cppclass InpaintFunctor(GCoptimizationGridGraph.SmoothCostFunctor):
    int f
    int w
    int h
    int n_labels
    int* image
    int* offsets
    int* known
    
    # Since I always get confused
    # x+y*width = site label
    # col + row * width 
    # channel + (x + y*width) * 3
    # offsets [:,0] row/y offset
    # offsets [:,1] col/x offset

    __init__(int f_, int w_, int h_, int n_labels_, int* image_, int* offsets_, int* known_):
        this.w = w_
        this.h = h_
        this.n_labels = n_labels_
        this.image = image_
        this.offsets = offsets_
        this.known = known_

    int imageIndexFromSubs(int x, int y, int c):
        return c + (x + y *this.w) * 3

    int is_known(int x, int y):
        return this.known[x + y * this.w] == 1

    int is_valid(int x, int y):
        return x >=0 and x < this.w and y >= 0 and y < this.h and is_known(x,y)
    
    int compute_seam(int s, int l1, int l2):
        cdef int x = s % this.w
        cdef int y = (s - x) / this.w
        
        cdef int x1 = x + this.offsets[1 + 2 * l1]
        cdef int y1 = y + this.offsets[0 + 2 * l1]

        cdef int x2 = x + this.offsets[1 + 2 * l2]
        cdef int y2 = y + this.offsets[0 + 2 * l2]
        
        # for destination pixels that are not known, bail with 0 energy
        # since single site infinity handles it
        if(not is_valid(x1,y1)):
            return 0
        if(not is_valid(x2,y2)):
            return 0
            
        cdef int c
        cdef int res
        cdef int t1
        cdef int t2
        res = 0
        for c in range(3):
            t1 = this.image[imageIndexFromSubs(x1,y1,c)]
            t2 = this.image[imageIndexFromSubs(x2,y2,c)]
            tmp = t1 - t2
            res += tmp * tmp
        return res
    
    int compute(int s1, int s2, int l1, int l2):
        # ||I(s1 + l1) - I(s1 + l2)||^2 + ||I(s2 + l1) - I(s2 + l2)||^2
        if(l1 == l2): return 0
        cdef int e1 = compute_seam(s1,l1,l2)
        cdef int e2 = compute_seam(s2,l1,l2)
        return e1 + e2

def cut_inpaint(np.ndarray[np.int32_t, ndim=3, mode='c'] unary_cost,
        np.ndarray[np.int32_t, ndim=2, mode='c'] offsets,
        np.ndarray[np.int32_t, ndim=4, mode='c'] images,
        np.ndarray[np.int32_t, ndim=4, mode='c'] knowns,
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
    if images.shape[1] != unary_cost.shape[0] and images.shape[2] != unary_cost.shape[1]:
        raise ValueError("unaray_cost shape must much image shape")
    if images.shape[3] != 3:
        raise ValueError("Image must be RGB")
    if images.shape[0] != knowns.shape[0] and images.shape[1] != knowns.shape[1] and images.shape[2] != knowns.shape[2]:
        raise ValueError("known shape must match image shape")

    # everything is ROW major at this point x = col, y = row

    cdef int h = unary_cost.shape[0]
    cdef int w = unary_cost.shape[1]
    cdef int n_labels = offsets.shape[0]

    cdef GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(w, h, n_labels)
    gc.setDataCost(<int*>unary_cost.data)
    gc.setSmoothCostFunctor(<InpaintFunctor*>new InpaintFunctor(w, h, n_labels, <int*>images.data, <int*>offsets.data, <int*>knowns.data))
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
    result_shape[0] = h
    result_shape[1] = w
    cdef np.ndarray[np.int32_t, ndim=2] result = np.PyArray_SimpleNew(2, result_shape, np.NPY_INT32)
    cdef int * result_ptr = <int*>result.data
    for i in xrange(w * h):
        result_ptr[i] = gc.whatLabel(i)
    return result