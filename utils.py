import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def generate_plots(list_of_dirs, titles, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for idx, yaxis in enumerate(['train_accs', 'valid_accs', 'train_losses', 'valid_losses']):
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        ax.set_title(titles[idx])
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """
    batch_size, num_class = logits.size()
    one_hot_labels = F.one_hot(labels, num_classes=num_class)

    # outputs = torch.exp(logits) / (torch.sum(torch.exp(logits), dim=1).view(-1, 1) + 1e-10)
    outputs = torch.softmax(logits, dim=1)
    loss = - torch.sum(torch.mul(one_hot_labels, torch.log(outputs + 1e-10)))    # add a small value to avoid numerical instability

    return loss / batch_size

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc


if __name__ == "__main__":
    logs_path = ['./train_log/mixer_p4_lr1', './train_log/mixer_p4_lr2', './train_log/mixer_p4', './train_log/mixer_p4_lr4', './train_log/mixer_p4_lr5']
    titles = ['MLPMixer training accuracy comparison', 'MLPMixer validation accuracy comparison', 'MLPMixer training loss comparison', 'MLPMixer validation loss comparison']
    legend_names = ['learning rate=0.1', 'learning rate=0.01', 'learning rate=0.001', 'learning rate=0.0001', 'learning rate=0.00001']
    generate_plots(logs_path, titles, legend_names, './')