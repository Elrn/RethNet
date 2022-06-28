import os
from os.path import join, dirname
from absl import flags, logging

# logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

set_log_verv = lambda debug:logging.set_verbosity(logging.DEBUG) if FLAGS.debug else logging.set_verbosity(logging.INFO)
base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = join(base_dir, 'log')

"""
flags.register_validator('flag',
                         lambda value: value % 2 == 0,
                         message='some message when assert on')
flags.mark_flag_as_required('is_training')
"""

########################################################################################################################
""" model setting """
########################################################################################################################
flags.DEFINE_boolean('predict_step', True, '간단한 prediciton의 경우 사용, 대용량 prediction의 경우 false')
flags.DEFINE_string('saved_model_name', 'SavedModel', 'Saved model folder Name')

########################################################################################################################
""" Training settings """
########################################################################################################################
flags.DEFINE_boolean('train', True, '모델 학습을 위한 모드')
flags.DEFINE_boolean('save', True, 'wether save the model after training')
flags.DEFINE_boolean('plot', True, 'wether plot prediction of the model.')
flags.DEFINE_integer("epoch", 100, "")
flags.DEFINE_float("lr", 0.001, "")

ckpt_file_name = 'EP_{epoch}, L_{loss:.0f}, P_{Precision:.3f}, R_{Recall:.3f}, F2_{F2_Score:.3f}, D_{DSC:.3f}, J_{JSC:.3f} ' \
                 'vL_{val_loss:.0f}, vP_{val_Precision:.3f}, vR_{val_Recall:.3f}, vF2_{val_F2_Score:.3f}, vD_{val_DSC:.3f}, vJ_{val_JSC:.3f}'\
                 '.h5'
flags.DEFINE_string('ckpt_file_name', ckpt_file_name, 'checkpoint file name')

########################################################################################################################
""" Dataset Setting """
########################################################################################################################
flags.DEFINE_integer("bsz", 19, "Batch size")
flags.DEFINE_integer("repeat", 1, "Batch size")
flags.DEFINE_float("valid_split", 0.1, "Batch size")

########################################################################################################################
""" Directory """
########################################################################################################################
flags.DEFINE_string('ckpt_dir', join(log_dir, 'checkpoint'), '체크포인트/모델 저장 경로')
flags.DEFINE_string('plot_dir', join(log_dir, 'plot'), 'plot 저장 경로')

########################################################################################################################
""" Predict """
########################################################################################################################
flags.DEFINE_multi_string('inputs', None, 'list paths for prediction')
# flags.DEFINE_multi_string('inputs', ['C:\dataset\stroke\\00632.nii.gz'], 'list paths for prediction')

########################################################################################################################