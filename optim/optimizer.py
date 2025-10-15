import torch
import torch.optim as optim

class GradClipWrapper:
    """
    在 optimizer.step() 前执行梯度裁剪的轻量包装器。
    支持：范数裁剪(max_norm) 和 值裁剪(clip_value)；两者都给则先值裁剪，再范数裁剪。
    """
    def __init__(self, optimizer, max_norm=None, clip_value=None):
        self._optim = optimizer
        self.max_norm = max_norm
        self.clip_value = clip_value

    # 透传常用接口/属性
    @property
    def param_groups(self):
        return self._optim.param_groups

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, sd):
        return self._optim.load_state_dict(sd)

    def zero_grad(self, *args, **kwargs):
        return self._optim.zero_grad(*args, **kwargs)

    def add_param_group(self, *args, **kwargs):
        return self._optim.add_param_group(*args, **kwargs)

    def step(self, closure=None):
        params = [p for g in self._optim.param_groups for p in g['params'] if p.grad is not None]
        if params:
            if self.clip_value is not None:
                torch.nn.utils.clip_grad_value_(params, self.clip_value)
            if self.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, self.max_norm)
        return self._optim.step(closure)


def build_optimizer(hparams):
    def make_optimizer(params):
        optim_method = hparams['optim_method'].lower()
        if optim_method == 'adam':
            base_opt = optim.Adam(
                params,
                lr=hparams['learning_rate'],
                weight_decay=hparams['weight_decay']
            )
        elif optim_method == 'sgd':
            base_opt = optim.SGD(
                params,
                lr=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                momentum=hparams['momentum']
            )
        else:
            raise NotImplementedError(f"Unknown optim_method: {optim_method}")

        # —— 可选裁剪超参（没有就不裁剪）——
        max_norm = float(hparams.get('grad_clip', 0) or 0)
        clip_value = hparams.get('grad_clip_value', None)
        if clip_value is not None:
            clip_value = float(clip_value)

        if (max_norm > 0) or (clip_value is not None):
            print(f"[GradClip] enabled: max_norm={max_norm if max_norm > 0 else None}, clip_value={clip_value}")
            return GradClipWrapper(base_opt,
                                   max_norm=max_norm if max_norm > 0 else None,
                                   clip_value=clip_value)
        return base_opt

    return make_optimizer