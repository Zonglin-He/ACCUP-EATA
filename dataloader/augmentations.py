import numpy as np
import torch
from utils.utils import fix_randomness

def DataTransform(sample, config, seed_id): # 根据数据集名称选择幅度扭曲参数，生成两种增强
    fix_randomness(seed_id) # 每次调用时固定随机种子，保证两种增强结果一致
    name = config.__class__.__name__

    if name == 'HAR':  # 原: sigma=0.1, knot=150
        aug = magnitude_warp(sample, sigma=0.05, knot=32)
        aug2 = magnitude_warp(sample, sigma=0.10, knot=16)
    elif name == 'EEG':  # 原: sigma=0.1, knot=30
        aug = magnitude_warp(sample, sigma=0.05, knot=16)
        aug2 = magnitude_warp(sample, sigma=0.10, knot=8)
    elif name == 'FD':  # 原: sigma=0.01, knot=150
        aug = magnitude_warp(sample, sigma=0.02, knot=32)
        aug2 = magnitude_warp(sample, sigma=0.05, knot=16)
    else:
        raise NameError
    return aug, aug2

def jitter(x, sigma = 0.04): # 抖动噪声增强：对输入数据每个值加入均值0、标准差sigma的高斯噪声
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
     # 输入 x 可为 numpy 或 tensor，若为tensor则先转numpy。返回添加噪声后的数组
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.05):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    factor = np.random.normal(1., sigma, size=(x.shape[0], x.shape[1], 1))  # (N,C,1)
    return x * factor

def window_slice(x, reduce_ratio=0.9): 
    # 窗口切片增强：随机裁剪并重新缩放时间轴长度。
    if not isinstance(x, np.ndarray): # 输入 x 可为 numpy 或 tensor，若为tensor则先转numpy
        x = x.cpu().numpy()
    x = x.transpose((0, 2, 1)) # 转为 (N, L, C) 形状，方便后续按时间轴处理
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int) # 计算裁剪后目标长度
     # 若目标长度大于等于原长度，则不裁剪，直接返回原数据
    if target_len >= x.shape[1]:
        return x.transpose((0, 2, 1))
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    # 对每个样本随机选择一个起点starts，在 [0, L-target_len) 范围，然后提取该区间长度target_len的序列片段
    ends = (target_len + starts).astype(int) # 计算每个样本的终点ends
    ret = np.zeros_like(x) # 预先分配一个 (N, L, C) 的结果数组
    for i, pat in enumerate(x): # 对每个样本 i，按起点和终点裁剪，并通过线性插值将裁剪片段缩放回原长度
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T #np.interp实现线性插值
    ret = ret.transpose((0, 2, 1)) # 转回 (N, C, L) 形状
    return ret

def magnitude_warp(x, sigma=0.1, knot=150):  # 幅度扭曲增强：通过随机曲线对时间序列幅值进行平滑拉伸/压缩
    from scipy.interpolate import CubicSpline
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy() 
    x = x.transpose((0, 2, 1)) # 转为 (N, L, C) 形状，方便后续按时间轴处理
    orig_steps = np.arange(x.shape[1]) # 生成时间步索引数组 [0, 1, ..., L-1]

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # 为每个样本和每个通道生成 knot+2 个随机幅度因子，服从均值1、标准差sigma的高斯分布
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    # 生成 knot+2 个均匀分布的时间步索引，形状为 (knot+2, 1)，用于后续插值
    ret = np.zeros_like(x) # 预先分配一个 (N, L, C) 的结果数组
    for i, pat in enumerate(x): # 对每个样本 i，按通道对随机幅度因子进行三次样条插值，生成与原时间步长度相同的平滑幅度曲线
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper # 将原始序列 pat 与平滑幅度曲线 warper 按时间点逐元素相乘，得到增强结果

    ret = ret.transpose((0, 2, 1)) # 转回 (N, C, L) 形状
    return ret


def time_warp(x, sigma=0.1, knot=30): # 时间扭曲增强：通过随机曲线对时间序列时间轴进行平滑拉伸/压缩
    # 与 magnitude_warp 类似的思路，但作用在时间坐标上
    from scipy.interpolate import CubicSpline
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    x = x.transpose((0, 2, 1))
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    # 生成 random_warps（与时间轴长度同knot+2点）后，不直接当缩放曲线用，而是生成一条扭曲的时间索引映射 time_warp
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    # 生成 knot+2 个均匀分布的时间步索引，形状为 (knot+2, 1)，用于后续插值
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            #用于将扭曲后的时间坐标拉伸到 [0, L-1] 范围
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    ret = ret.transpose((0, 2, 1))
    return ret #最终效果是序列的某些部分被放慢或加速（时间轴扭曲），然后再映射回原长度，产生非线性时间畸变的序列

def permutation(x, max_segments=5, seg_mode="random"): # 顺序排列增强：将时间序列随机分段并重排顺序
     # max_segments=5 表示最多切分成5段
    orig_steps = np.arange(x.shape[2]) # 生成时间步索引数组 [0, 1, ..., L-1]
     # 输入 x 可为 numpy 或 tensor，若为tensor则先转numpy
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1: #对每个样本，随机生成一个1~max_segments 的段数 num_segs[i]
            if seg_mode == 'random':
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.array_split(orig_steps, num_segs[i])
                # 如果 seg_mode=='random'，则随机选择 num_segs[i]-1 个分割点，将序列分割成若干段
            else:
                splits = np.array_split(orig_steps, num_segs[i])
                # 否则平均分成 num_segs 段
            warp = np.concatenate(np.random.permutation(splits)).ravel() # 随机打乱各段顺序并重新拼接
            ret[i] = pat[:, warp] # 按打乱后的时间步索引 warp 重排原序列
        else:
            ret[i] = pat # 若只分成1段，则不变

    return torch.from_numpy(ret) # 返回重排后的结果（tensor格式）

import math
def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None, distribution='geometric'):
    # 用mask遮蔽部分时间步以生成对比学习的视图
    # 输入 sample (N, C, L)，masking_ratio表示要mask掉的时间比例，lm决定mask片段平均长度
    # positive_nums表示生成多少个增强视图，distribution表示mask片段长度分布
    if positive_nums is None: #positive_nums 未指定则依据masking_ratio计算（确保覆盖一定比例）
        positive_nums = math.ceil(1.5 / (1 - masking_ratio)) 

    sample = sample.permute(0, 2, 1)
    sample_repeat = sample.repeat(positive_nums, 1, 1)
    # 将sample转为 (N, L, C)，再复制positive_nums次（得到多个增强视图）
    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution) # 生成与sample_repeat同形状的mask
    x_masked = mask.float() * sample_repeat # 用mask遮蔽部分时间步（mask处为0）

    return x_masked.permute(0, 2, 1), mask.permute(0, 2, 1) # 返回遮蔽后的结果和mask（均转回 (N, C, L)）


def geom_noise_mask_single(L, lm, masking_ratio): #几何分布掩码生成：按给定平均长度lm产生交替的mask段和保留段
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None): 
    # 生成与输入 X 相同形状的bool掩码，False表示需要mask的位置
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension

            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask).to(X.device)