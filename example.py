import pygco3d
import pygco
import numpy as np


unary_cost = np.load('data/unary_cost.npy')
d_offsets = np.load('data/d_offsets.npy')
imges = np.load('data/imges.npy')[:1]
masks = np.load('data/masks.npy')[:1]
unary_cost = np.random.randint(60, size=(unary_cost.shape))


def my():
    labels = pygco3d.cut_inpaint(unary_cost.astype(np.int32),
                                 d_offsets.astype(np.int32).copy(order='C'),
                                 imges.astype(np.int32),
                                 (~masks).astype(np.int32),
                                 n_iter=-1,
                                 algorithm='swap',
                                 verbosity=1)


def other():
    labels = pygco.cut_inpaint(unary_cost.astype(np.int32),
                               d_offsets.astype(np.int32).copy(order='C'),
                               imges[0].astype(np.int32),
                               (~masks)[0].astype(np.int32),
                               n_iter=-1,
                               algorithm='swap',
                               verbosity=1)


other()
