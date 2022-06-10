import numpy as np
import tensorflow as tf
import os, logging, re
from absl import app
import nibabel as nib
from Dataset import skin_lesion
from os.path import join
import flags, utils, train
FLAGS = flags.FLAGS

########################################################################################################################
""" 전역변수 설정 """
########################################################################################################################
data = skin_lesion

########################################################################################################################
def main(*argv, **kwargs):
    """  """
    """ init """
    utils.tf_init()
    paths = [FLAGS.ckpt_dir, FLAGS.plot_dir]
    [utils.mkdir(path) for path in paths]
    if FLAGS.train:
        train.main(argv, _dataset=data)
    else: # inference
        saved_model_path = join(FLAGS.ckpt_dir, FLAGS.saved_model_name)
        model = tf.saved_model.load(saved_model_path)

        from glob import glob
        # FLAGS.inputs = glob('C:\dataset\stroke\ADC2dwi\*')
        # FLAGS.inputs = glob('C:\dataset\stroke\dwi_RPI_BFC\*')
        FLAGS.inputs = glob('C:\dataset\\01_KUMC_data\\*\*\\*dwi_RPI_BFC*')

        ds, depths, headers, affines, img_shapes = data.build_for_pred(FLAGS.inputs)
        ds = ds.as_numpy_iterator() # load
        # OOM 을 피하기 위해 split 하여 입력
        outputs = []
        for arr in ds:
            output = model(arr)
            outputs.append(output)
        outputs = np.concatenate(outputs, 0)
        """ output post-processing """
        data.post_processing(outputs, depths, headers, affines, img_shapes)

########################################################################################################################
if __name__ == '__main__':
    app.run(main)

########################################################################################################################





