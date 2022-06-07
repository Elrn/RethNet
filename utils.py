import os, datetime, re, logging
import matplotlib.pyplot as plt
import numpy as np
# import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf

########################################################################################################################
""" LOGGING """
# logging.debug(f'') logging.info(f'') logging.warning(f'') logging.error(f'')
format = '[%(asctime)s]|[%(name)s]|[%(levelname)s] %(message)s'
formatter = logging.Formatter(format)
logging.basicConfig(
    # filename='log.txt',
    format=format,
    datefmt='%m/%d/%Y %I:%M:%S',
    level=logging.DEBUG
)
def get_logger(name=None, level=None):
    level = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL][level]
    logger = logging.getLogger(f"{name}")
    if level is not None:
        logger.setLevel(level)

    return logger

# stream_hander = logging.StreamHandler()
# stream_hander.setFormatter(formatter)
# logger.addHandler(stream_hander)
########################################################################################################################
def tf_init():
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    logging.getLogger('h5py._conv').disabled = True

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

########################################################################################################################
safe_divide = lambda a, b : np.divide(a, b, out=np.zeros_like(a), where=b!=0)
to_list = lambda x: [x] if type(x) is not list else x
get_datetime = lambda :datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
########################################################################################################################
def mkdir(path):
    try: # if hasattr(path, '__len__') and type(path) != str:
        os.makedirs(path)
    except OSError as error:
        print(error)

# date = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')

#
def save_history(history, path):
    metrics = list(history.history)
    len_metrics = int(len(metrics) // 2)
    fig, ax = plt.subplots(1, len_metrics, )
    # ax = ax.ravel()  # flatten
    for i in range(len_metrics):
        ax[i].plot(history.history[metrics[i]])
        ax[i].plot(history.history["val_" + metrics[i]])
        ax[i].set_title(f"Model {metrics[i]}")
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metrics[i])
        ax[i].legend(["train", "val"])
    fig.tight_layout()  # fix overlapping between title and labels
    fig.savefig(f'{path}', dpi=300)

########################################################################################################################
# DataSet Utils
"""
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
"""
########################################################################################################################
def SWN(image, level=50, window=250, dtype='float32'):
    """ stochastic tissue window normalization
    https://arxiv.org/abs/1912.00420

    # value = {'brain':        [40, 80], * 10?!
    #          'lungs':        [-600, 1500],
    #          'liver':        [30, 150],
    #          'Soft tissues': [50, 250],
    #          'bone':         [400, 1800],
    #          }
    """
    def preprocessing(x):
        if type(x) == list or type(x) == tuple:
            return x[0], x[1]
        elif type(x) == int:
            return x, 0
        else:
            raise TypeError(f'Input type must "list", "tuple" or "int" but "{type(x)}".')

    image = tf.cast(image, dtype)
    level_mean, level_std = preprocessing(level)
    window_mean, window_std = preprocessing(window)

    level = tf.random.normal([1], level_mean, level_std)
    window = tf.random.normal([1], window_mean, window_std)
    window = tf.math.abs(window)

    max_threshold = level + window / 2
    min_threshold = level - window / 2

    image = tf.clip_by_value(image, min_threshold, max_threshold)
    (image - min_threshold) / (max_threshold - min_threshold)
    return image


def read_medical_file(file_path, ):
    # _, ext = os.path.splitext(file_path)
    print(os.path.basename(file_path))
    image = sitk.ReadImage(file_path, sitk.sitkFloat64)  # channel first / i.e sitk.sitkInt16
    ndarray = sitk.GetArrayFromImage(image)
    ndarray = np.transpose(ndarray, [1, 2, 0])  # make channel last
    # if ext == 'dcm':
    #     # ndarray = dcm.read_file(file_path).pixel_array # return non-channel
    #     image = sitk.ReadImage(file_path, sitk.sitkFloat64)  # channel first / i.e sitk.sitkInt16
    #     ndarray = sitk.GetArrayFromImage(image)
    #     ndarray = np.transpose(ndarray, [1, 2, 0])  # make channel last
    # elif ext == 'nii' or 'nii.gz' in os.path.basename(file_path) :
    #     ndarray = nib.load(file_path).get_fdata() # channel last
    # else:
    #     raise ValueError(f"Unexpected Medical file's extension, [{file_path}]")
    if not isinstance(ndarray, np.ndarray):
        raise ValueError(f'Type of Image must "ndarray", but "{type(ndarray)}" has returned.')
    return ndarray


def resampling(image, spacing=None, default_value=0):
    """
    sitk file의 spacing을 변환, 적용
    image size와 origin을 수정
    """
    img_sz = list(image.GetSize())

    if type(spacing) == float:
        spacing = [spacing] * len(img_sz)
    if type(spacing) == int:
        spacing = [float(spacing)] * len(img_sz)
    elif type(spacing) == list:
        if len(spacing) != len(img_sz):
            raise ValueError(f'')
    else:
        raise ValueError(f'')

    resize_rate = np.divide(list(image.GetSpacing()), spacing)
    target_sz = np.multiply(img_sz, resize_rate).astype('int32').tolist()

    print(f'[!] Original Image size "{img_sz}" resized to "{target_sz}".')

    def get_origin(image, spacing, size):
        sz = np.array(image.GetSize())
        sp = np.array(image.GetSpacing())
        new_size_no_shift = np.int16(np.ceil(sz * sp / spacing))
        origin = np.array(image.GetOrigin())
        shift_amount = np.int16(np.floor((new_size_no_shift - size) / 2)) * spacing
        return origin + shift_amount

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    resample.SetOutputSpacing(spacing)
    resample.SetSize(target_sz)
    resample.SetOutputOrigin(get_origin(image, spacing, target_sz))

    return resample.Execute(image)


########################################################################################################################
def none_zeros_coord(image):
    """
    ndarray 내 0이 아닌 값들의 좌표값을 반환
    """
    mask = image > 0
    return np.argwhere(mask)

def make_paddings(diff):
    diff *= diff > 0
    paddings = np.stack([(diff+1)//2, diff//2], 1)
    assert False not in (np.sum(paddings, -1) == diff)
    return paddings

def crop_voxel(image, tartget_size, crop_ratio=None, use_zero_crop=None):
    """
    crop_ratio = 각 축별 양옆 crop 역역 할당 weight 값
    """
    shape = np.array(image.shape)
    tartget_size = np.array(tartget_size)
    assert len(shape) == len(tartget_size)
    assert len(tartget_size) == 3
    need_crop = (image.shape - tartget_size)

    def solve_around_problem(ratio):
        """
        np 반올림은 .5 를 버린다. 홀수를 반으로 나누면 .5로 나누어지므로 일부는 올리고 일부는 버린다.
        """
        ratio = np.transpose(ratio)
        # ratio를 적용 후 정수로 나누어 질 때 float 처리 한계로 완벽한 정수가 아닌 값이 나옴
        # 따라서 반올림으로 나머지를 버려야 제대로 된 정수를 얻을 수 있다.
        # floor 부분은 어차피 버리기 때문에 상관없음
        ratio[0] = np.round(ratio[0], 2)
        ratio[0] = np.ceil(ratio[0])
        ratio[1] = np.floor(ratio[1])
        ratio = np.transpose(ratio)
        return ratio

    def index_translate(ratio, img_shape):
        """
        crop할 양을 indexing 으로 치환,
        -0 은 이미지 size로 치환
        """
        ratio = np.transpose(ratio)
        ratio[1] *= -1
        for i, value in enumerate(ratio[1]):
            if value == 0:
                ratio[1][i] = img_shape[i]
        return ratio

    #### Crop zero padding Block ####
    if use_zero_crop == True:
        coords = none_zeros_coord(image)
        check_margin = np.stack([coords.min(0), shape - (coords.max(0) + 1)], 1)  # 양방향 여유
        total_margins = np.sum(check_margin, 1)
        margin_ratio = np.divide(check_margin, np.stack([total_margins, total_margins], 1))
        margin_ratio = np.nan_to_num(margin_ratio)
        crop_margin = total_margins - need_crop
        zero_crop = np.where(crop_margin > 0, need_crop, total_margins)
        need_crop -= zero_crop
        amount_margin_crop = margin_ratio * np.stack([zero_crop, zero_crop], 1)
        amount_margin_crop = solve_around_problem(amount_margin_crop).astype(np.int32)
        index_translate(amount_margin_crop, image.shape)

        image = image[
                amount_margin_crop[0][0]:amount_margin_crop[0][1],
                amount_margin_crop[1][0]:amount_margin_crop[1][1],
                amount_margin_crop[2][0]:amount_margin_crop[2][1]]

    #### Crop Image Block ####
    amount_image_crop = need_crop
    if crop_ratio is not None:
        # 단순 [1, 0] 으로 할당할 경우 image 에 맞게 broadcasting
        if len(np.array(crop_ratio)) == 2:
            crop_ratio = np.tile(crop_ratio, [len(shape), 1])
        if np.any(np.sum(crop_ratio, 1) > 1):
            raise ValueError(f"crop_ratio's sum MUST under '1', but got {np.sum(crop_ratio, 1)}")
        assert len(shape) == len(np.array(crop_ratio))
        amount_image_crop = np.stack([amount_image_crop, amount_image_crop], 1).astype(float)
        amount_image_crop *= crop_ratio
        amount_image_crop = solve_around_problem(amount_image_crop)
    else:
        amount_image_crop = np.stack([amount_image_crop + 1, amount_image_crop], 1) // 2
    amount_image_crop = amount_image_crop.astype(np.int32)
    index_translate(amount_image_crop, image.shape)

    image = image[
            amount_image_crop[0][0]:amount_image_crop[0][1],
            amount_image_crop[1][0]:amount_image_crop[1][1],
            amount_image_crop[2][0]:amount_image_crop[2][1]]

    print(f'[!] Image Cropped "{need_crop}", Original size "{shape.tolist()}" to "{image.shape}".')
    return image

def crop_n_pad(image, size, axis=None, crop_ratio=None, use_zero_crop=None):
    img_shape = np.array(image.shape)
    size = list(size)
    if axis == None:
        # size 만 주어질 경우 image rank 에 맞는 size rank 와 단일 size를 받는 경우를 구분
        if len(size) == 1 and len(size) != len(img_shape):
            size *= len(img_shape)
        if len(img_shape) != len(size):
            raise ValueError(f'')
    else:
        # size rank와 axis rank가 일치해야 하며, size를 image rank에 맞게 변환
        axis = np.array(axis)
        if len(np.array(size)) != len(axis):
            raise ValueError(f'')
        if len(axis) != 3:
            axis_to_dim = {ax: size[i] for i, ax in enumerate(axis)}
            size = [axis_to_dim[i] if i in axis else img_shape[i] for i in range(len(img_shape))]

    # [Padding Block]
    paddings = make_paddings(size - img_shape)
    padded_image = np.pad(image, paddings) #
    print(f'[!] Image Padded "{paddings.tolist()}", Original size "{image.shape}" to "{padded_image.shape}"')

    # [Cropping Block]
    # arr내 zero 가 불균형하게 존재할 경우 필요.
    # zero 값들을 우선 고려하여 일부 혹은 전부 crop 후 남은 crop을 진행.
    if use_zero_crop == True:
        padded_image = crop_voxel(padded_image, size, crop_ratio, use_zero_crop=use_zero_crop)
    else:
        padded_image = crop_voxel(padded_image, size, crop_ratio)

    return padded_image


def crop_zeros(image):
    rank = len(image.shape)
    if rank != 2 and rank != 3:
        raise ValueError(f'Image Rank must 2 or 3, but {rank}.')
    # 0이 아닌 값들의 처음과 끝
    coords = none_zeros_coord(image)
    min, max = coords.min(0), coords.max(0) + 1

    image = image[min[0]:max[0], min[1]:max[1]]
    if rank == 3:
        image = image[:, :, min[2]: max[2]]
    return image


def count_patches(input_shape, sizes, strides, paddings='VALID'):
    """
    : Need to reflect variables to number of patches along paddings
        XY(Z) 축만을 필요,
    extract patches 후 개수 혹은 Convolution 후 output image size 계산
    """
    assert len(sizes)==len(strides)
    to_arr = lambda x:np.array(x)
    shape = (to_arr(input_shape) - to_arr(sizes)) / to_arr(strides) + 1
    return shape.astype(int)

def reconstruct_pathces(input, batch_size, origin_size, ksize, strides):
    """
    : patch size 와 strides 가 일치하지 않을 때 reconstruction 을 반영해야 한다.
        multi-channel 고려 필요!
    """
    # except batch/channel dimension
    num_patches_along_dims = count_patches(origin_size, ksize, strides)[1:-1]
    # batch 병렬처리를 위해 batch dimension을 추가
    input = tf.stack(tf.split(input, batch_size, 0), 0)
    # input = tf.reshape(input, [batch_size, x.shape[0]//batch_size]+input.shape[1:]])
    split_dim = 1
    for axis, num_patches in enumerate(reversed(num_patches_along_dims)):
        if num_patches == 1: continue # patch 가 1개라서 concat 할 대상이 없음
        split = tf.split(input, input.shape[split_dim]//num_patches, split_dim)
        # axis 는 extract patches 의 extract dimension 순서에 따라 기입
        concat_dim = -(axis+2) # Z > Y > X
        split = [tf.concat(tf.split(x, x.shape[split_dim], split_dim), concat_dim) for x in split]
        """ for 문을 활용하기 위한 코드 
        위 코드 까지 실행 시 split 으로 나눠진 행렬을 loop 로 돌려야하는데,
        처음 실행 조건은 batch dimension 에 모여진 patch 들을 다루기 때문에 
        split 으로 나눠져 list 에 나눠줘 있는 patch 들을 다시 batch dimension 으로 합산
        """
        perm = [i for i in range(tf.rank(split))][3:]
        input = tf.squeeze(tf.transpose(split, [1,2,0]+perm), split_dim)
    # split dimension 을 제거
    output = tf.squeeze(input, axis=split_dim)
    return output


########################################################################################################################