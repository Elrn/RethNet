from tensorflow.python.keras.callbacks import *
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import utils
import numpy as np
import re
import logging
from tensorflow.python.keras import backend
from keras.utils import io_utils
import flags
FLAGS = flags.FLAGS
from pathlib import Path

########################################################################################################################
class log(tf.keras.callbacks.Callback):
    def __init__(self, file_path, ext='.csv'):
        super(log, self).__init__()
        self.file_path = file_path + ext # 확장자는 callback 내에서 처리?

    def on_epoch_end(self, epoch, logs=None):
        epoch = {'epoch': epoch} # scalar to dict
        logs = {**epoch, **logs}

        is_key = Path(self.file_path).is_file()
        f = open(self.file_path, 'a')
        w = csv.writer(f)
        if is_key != True:
            w.writerow(logs.keys())
        w.writerow(logs.values())
        f.close()

########################################################################################################################
class monitor(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, dataset=None, num_show=30, fig_size_rate=3, name=None):
        super(monitor, self).__init__()
        self.save_dir = save_dir
        self.dataset = dataset.take(num_show // FLAGS.bsz + 1)
        self.num_show = num_show
        self.fig_size_rate = fig_size_rate
        self.name = name
    #
    def on_epoch_end(self, epoch, logs=None):
        self.plot(epoch, logs=None)

    def plot(self, epoch, logs=None):
        """
        col 1. 원본
        col 2. label
        col 3. 원본 + label
        col 4. 원본 + prediction
        """
        epoch += 1
        ylabels = ['Data', 'Lable', 'Data + Lable', 'Prediction', 'Data + Prediction']
        cols, rows = len(ylabels), self.num_show
        figure, axs = plt.subplots(cols, rows, figsize=(rows * 3, cols * 3))
        figure.suptitle(f'Epoch: "{epoch}"', fontsize=10)
        figure.tight_layout()

        [axs[i][0].set_ylabel(l, fontsize=14) for i, l in enumerate(ylabels)]

        inputs, preds, labels = [], [], []
        for data, seg in self.dataset: # take, B, H, W, C
            inputs.append(data)
            preds.append(self.model(data))
            labels.append(seg)

        """
        병렬 처리를 하지 않을 경우 코드는 더 간결할 수 있다.
        대신, Batch size 를 1로 설정하고 모델에 각각 입력해야 한다.
        """
        inputs = tf.squeeze(inputs)
        labels = tf.squeeze(tf.argmax(labels, -1))
        preds = tf.squeeze(tf.argmax(preds, -1))
        inputs = tf.reshape(inputs, [-1, *inputs.shape[-3:]])
        labels = tf.reshape(labels, [-1, *labels.shape[-2:]])
        preds = tf.reshape(preds, [-1, *preds.shape[-2:]])

        for c in range(cols):
            for r in range(rows):
                axs[c][r].yaxis.tick_right()
                # axs[c][r].set_xticks([])
                # axs[c][r].set_yticks([])
                if c == 0:
                    axs[c][r].imshow(labels[r], cmap='rainbow', alpha=0.5)

                elif c == 1:
                    axs[c][r].imshow(inputs[r])
                    axs[c][r].imshow(labels[r], cmap='rainbow', alpha=0.2)
                elif c == 2:
                    axs[c][r].imshow(inputs[r])
                elif c == 3:
                    axs[c][r].imshow(preds[r], cmap='rainbow', alpha=0.5)
                else:
                    axs[c][r].imshow(inputs[r])
                    axs[c][r].imshow(preds[r], cmap='rainbow', alpha=0.2)

        filename = f'{epoch}.png' if self.name == None else self.name + f'-{epoch}.png'
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=200)
        plt.close('all')

    def get_cmap(self):
        transparent = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
        white = matplotlib.colors.colorConverter.to_rgba('y', alpha=0.5)
        red = matplotlib.colors.colorConverter.to_rgba('r', alpha=0.7)
        return matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [transparent, white, red], 256)


########################################################################################################################
class load_weights(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(load_weights, self).__init__()
        self.filepath = os.fspath(filepath) if isinstance(filepath, os.PathLike) else filepath

    def on_train_batch_begin(self, logs=None):
    # def on_predict_batch_begin(self, logs=None):
    # def on_predict_begin(self, logs=None):
        self.load_weights()

    def load_weights(self):
        filepath_to_load = (self._get_most_recently_modified_file_matching_pattern(self.filepath))
        if (filepath_to_load is not None and self.checkpoint_exists(filepath_to_load)):
            try:
                self.model.load_weights(filepath_to_load)
                print(f'[!] Saved Check point is restored from "{filepath_to_load}".')
            except (IOError, ValueError) as e:
                raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')

    @staticmethod
    def checkpoint_exists(filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return tf.io.gfile.exists(filepath)
        tf_saved_model_exists = tf.io.gfile.exists(filepath)
        tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
            filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists

    @staticmethod
    def _get_most_recently_modified_file_matching_pattern(pattern):
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if tf.io.gfile.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name

########################################################################################################################
class setLR(Callback):
    def __init__(self, lr, verbose=0):
        super(setLR, self).__init__()
        self.lr = lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if not isinstance(self.lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             f'should be float. Got: {self.lr}')
        if isinstance(self.lr, tf.Tensor) and not self.lr.dtype.is_floating:
            raise ValueError(
                f'The dtype of `lr` Tensor should be float. Got: {self.lr.dtype}')
        backend.set_value(self.model.optimizer.lr, backend.get_value(self.lr))
        if self.verbose > 0:
            logging.info(
                f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning '
                f'rate to {self.lr}.')
            logs = logs or {}
            logs['lr'] = backend.get_value(self.model.optimizer.lr)

########################################################################################################################
# plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# early_stopping = EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False
# )
# ckpt = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=0, save_best_only=False,
#     save_weights_only=False, mode='auto', save_freq='epoch',
# )

