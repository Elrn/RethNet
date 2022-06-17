import tensorflow as tf
from os.path import join
from glob import glob
import flags
FLAGS = flags.FLAGS

#######################################################################################################################
base_dir = 'c:\\dataset\\skin lesion\\1\\'
data_dir = join(base_dir, '원본', '*.jpg')
seg_dir = join(base_dir, 'grayscale', '*.png')

data_paths = glob(data_dir)
seg_paths = glob(seg_dir)
# img_size = [480, 320]
img_size = [120, 80]
input_shape = [*img_size, 3]
seg_shape = [*img_size, 5]
num_class = 5
rank = 2
target_label = 4 # Metric 인자로 사용

"""
0:"background",
1:"hair",
2:"eyebrow, eye, mouth",
3:"normal",
4:"abnormal",
"""

#######################################################################################################################
def parse_fn(x, y):
    def read_img(path, channels):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, img_size, method='bilinear')
        return img

    def normalization(x):
        x -= tf.reduce_min(x, keepdims=True)
        x /= tf.reduce_max(x, keepdims=True)
        return x

    ### data
    x = read_img(x, 3)
    contrast_factor = tf.random.normal([], mean=3.5, stddev=1.0, dtype=tf.float32)
    x = tf.image.adjust_contrast(x, contrast_factor)
    x = normalization(x)
    x = tf.cast(x, tf.float32)
    tf.ensure_shape(x, input_shape)

    ### label
    y = read_img(y, 1)
    y = tf.squeeze(y, -1)
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(y, num_class, axis=-1)
    tf.ensure_shape(y, seg_shape)
    return x, y

def build(batch_size, validation_split=None):
    assert len(data_paths) == len(seg_paths)
    if validation_split != None and validation_split != 0:
        assert 0 < validation_split <= 0.5
        val_count = int(len(data_paths) * validation_split)
        buffer_size = (len(data_paths) - val_count) * FLAGS.repeat

        paths_set = (data_paths[val_count:], seg_paths[val_count:])
        dataset = load(paths_set, batch_size, buffer_size)

        paths_set = (data_paths[:val_count], seg_paths[:val_count])
        val_dataset = load(paths_set, batch_size)
        print(f'Validation dataset length: {val_count} of Total length: {len(data_paths)}')
        return dataset, val_dataset
    else:
        dataset = load((data_paths, seg_paths), batch_size)
        return dataset

def load(paths, batch_size, buffer_size=None, drop=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        paths
    ).prefetch(
        tf.data.experimental.AUTOTUNE
    # ).interleave(
    #     lambda x : tf.data.Dataset(x).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    #     cycle_length = tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls = tf.data.experimental.AUTOTUNE
    # ).repeat(
    #     count=FLAGS.repeat
    # ).cache(
    ).map(
        map_func=parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(
        batch_size=batch_size,
        drop_remainder=drop,
    )
    if buffer_size != None:
        print(f'[Dataset][Shuffle] buffer_size = {buffer_size}')
        dataset = dataset.shuffle(
            buffer_size,
            reshuffle_each_iteration=True
        )
    return dataset

#######################################################################################################################