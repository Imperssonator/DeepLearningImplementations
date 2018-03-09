import os
import numpy as np
import tensorflow as tf
from skimage import io


FLAGS = tf.app.flags.FLAGS


def save_image(data, data_format, e, suffix=None):
    """Saves a picture showing the current progress of the model"""

    X_G, X_real = data

    Xg = X_G[:8]
    Xr = X_real[:8]
    
    for i,gi in enumerate(Xg):
        io.imsave(os.path.join(FLAGS.fig_dir,
                               "epoch{}_sample{}_gen.png".format(e,i)),
                  gi)
    
    for i,ri in enumerate(Xr):
        io.imsave(os.path.join(FLAGS.fig_dir,
                               "epoch{}_sample{}_real.png".format(e,i)),
                  ri)


def get_stacked_tensor(X1, X2):

    X = tf.concat((X1[:16], X2[:16]), axis=0)
    list_rows = []
    for i in range(8):
        Xr = tf.concat([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
        list_rows.append(Xr)

    X = tf.concat(list_rows, axis=1)
    X = tf.transpose(X, (1,2,0))
    X = tf.expand_dims(X, 0)

    return X
