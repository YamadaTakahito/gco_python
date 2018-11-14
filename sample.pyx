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
    
    # Since I always get confused
    # x+y*width = site label
    # col + row * width 
    # channel + (x + y*width) * 3
    # offsets [:,0] row/y offset
    # offsets [:,1] col/x offset

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

        cdef int f1 = f + this.offsets[0 + 2 * l1]
        cdef int h1 = h + this.offsets[1 + 2 * l1]
        cdef int w1 = w + this.offsets[2 + 2 * l1]

        cdef int f2 = f + this.offsets[0 + 2 * l2]
        cdef int h2 = h + this.offsets[1 + 2 * l2]
        cdef int w2 = w + this.offsets[2 + 2 * l2]
        
        # for destination pixels that are not known, bail with 0 energy
        # since single site infinity handles it
        if not is_valid(f1, h1, w1):
            return 0
        if not is_valid(f2, h2, w2):
            return 0
            
        cdef int c # color
        cdef int error
        cdef int i1
        cdef int i2
        cdef int idx1 = f1 * (this.height * this.width * 3) + h1 * (this.width * 3) + w1 * 3
        cdef int idx2 = f2 * (this.height * this.width * 3) + h2 * (this.width * 3) + w2 * 3
        c = 0

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