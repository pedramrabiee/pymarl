from copy import deepcopy
import torch
import datetime
import os.path as osp
import os, glob


def hard_copy(source, requires_grad=False):
    copied = deepcopy(source)
    # set requires_grad
    for p in copied.parameters():
        p.requires_grad = requires_grad
    return copied


@torch.no_grad()
def polyak_update(target, source, tau):
    for p_targ, p in zip(target.parameters(), source.parameters()):
        p_targ.data.mul_(1 - tau)
        p_targ.data.add_(tau * p.data)

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def torchify(x, dtype=torch.float32, device='cpu', requires_grad=False):
    return torch.tensor(x,
                        requires_grad=requires_grad,
                        dtype=dtype,
                        device='cuda' if device == 'gpu' else 'cpu')


def to_device(x, device):
    # TODO: check if it is no-ops when the device is the same
    if device == 'gpu':
        return x.cuda()
    if device == 'cpu':
        return x.cpu()


def get_ac_space_info(ac_space):
    """returns action space shape"""
    # TODO add multidiscrete option here
    from gym.spaces import Box
    if isinstance(ac_space, Box):  # Continuous
        is_discrete = False
        return ac_space.shape[0], is_discrete
    else:  # Discrete
        is_discrete = True
        return ac_space.n, is_discrete


def get_save_checkpoint_name(run_root_dir):
    filename = 'ckpt_%s.pt' % get_timestamp(for_logging=False)
    path = osp.join(run_root_dir, 'checkpoints')
    os.makedirs(path, exist_ok=True)
    return osp.join(path, filename)

def get_load_checkpoint_name(current_root, load_run_name, timestamp):
    # change run name in the current run name to load_run_name, keep everything else the same
    path_split = current_root.split(os.sep)
    path_split[-2] = load_run_name
    path = osp.join(os.sep, *path_split, 'checkpoints')
    timestamp = get_last_timestamp(path) if timestamp == 'last' else timestamp
    filename = 'ckpt_%s.pt' % timestamp
    return osp.join(path, filename)

def get_last_timestamp(ckpt_dir):
    os.chdir(ckpt_dir)
    ckpt_list = glob.glob("*.pt")
    last_ckpt = ckpt_list[-1]
    last_ckpt_timestamp = last_ckpt[5:-3]  # checkpoint names ars in the format ckpt_YYYYMMDD_HHMMSS.pt
    return last_ckpt_timestamp

def get_timestamp(for_logging=True):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S') if for_logging else now.strftime('%Y%m%d_%H%M%S')
    timestamp_prefix = "%s | " % timestamp if for_logging else timestamp
    return timestamp_prefix

def n_last_eval_video_callable(n, value):
    def video_callable(x):
        return x % value in range(value-n, value)
    return video_callable


