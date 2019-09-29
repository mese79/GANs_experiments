import os
import sys
from time import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torchvision.utils import save_image

from data import sample_noise
from models import (DiscriminatorMNIST, GeneratorMNIST)
from losses import (
    standard_d_loss,
    standard_g_loss,
    heuristic_g_loss
)
from utility import(
    clear_line
)


SEED = 13
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(argv=[]):
    # config experiment
    num_epochs: int = 100000
    minibatch_size: int = 50
    d_learning_rate: float = 0.0002
    g_learning_rate: float = 0.0002
    discriminator_optim: str = 'adam'
    generator_optim: str = 'adam'
    loss_type: str = 'hack'
    z_dim: int = 100
    data_dim: int = 28 * 28
    d_hidden_size: int = 1024
    g_hidden_size: int = 256
    progress_update_interval = 20
    save_figs: bool = False
    save_model: bool = False
    if len(argv) > 1:
        save_model = bool(int(argv[1]))
        save_figs = save_model

    experiment_info = f'\ntotal iters: {num_epochs},\nbatch_size: {minibatch_size},\nd_lr: {d_learning_rate},\n' + \
                      f'g_lr: {g_learning_rate},\nloss: {loss_type},\nd_hidden_size: {d_hidden_size},\n' + \
                      f'g_hidden_size: {g_hidden_size},\ndisc_optim: {discriminator_optim},\n' + \
                      f'gen_optim: {generator_optim},\ndata_dim: {data_dim},\nz_dim: {z_dim},\n' + \
                      f'random seed: {SEED}'
    print(experiment_info)

    # Create experiment folders(if necessary).
    experiment_dir = Path(f'./experiments/mnist_exp_{time():.0f}')
    if save_model:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        fig_shots_dir = experiment_dir.joinpath('graph_shots')
        if save_figs:
            fig_shots_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment info file(if necessary).
    info_file = None
    if save_model:
        info_file = open(experiment_dir.joinpath('info.txt'), mode='w', encoding='utf-8')
        info_file.write(experiment_info)

    # plots
    fig, ax_data, ax_loss, ax_disc = prepare_plots(experiment_info, 'GANs - MNIST')
    plt.show(block=False)

    # MNIST Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, ), std=(0.5, ))]
    )

    train_dataset = datasets.MNIST(
        root='./datasets/mnist_data/', train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root='./datasets/mnist_data/', train=False, transform=transform, download=False
    )

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=minibatch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=minibatch_size, shuffle=False
    )

    # Creating device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\nrunning on: {device}')

    # Creating models
    D = DiscriminatorMNIST(data_dim, d_hidden_size).to(device)
    G = GeneratorMNIST(z_dim, g_hidden_size, data_dim).to(device)
    print(f'\n\n{D}\n\n{G}\n\n')

    if save_model:
        info_file.write(f'\nrunning on: {device}')
        info_file.write(f'\n\n{D}\n\n{G}\n\n\n')
        info_file.flush()

    if discriminator_optim == 'sgd':
        d_optimizer = torch.optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.65)
    else:
        d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))

    if generator_optim == 'sgd':
        g_optimizer = torch.optim.SGD(G.parameters(), lr=g_learning_rate, momentum=0.65)
    else:
        g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

    show_separate_loss = False
    d_real_loss_list: list = []
    d_fake_loss_list: list = []
    g_loss_list: list = []
    d_x_list: list = []
    d_g_z_list: list = []

    #
    # training loop
    #
    it = 0
    t1 = time()
    for epoch in range(1, num_epochs + 1):
        
        for batch_idx, (real_data, _) in enumerate(train_loader):
            it += 1
            # Training discriminator
            real_data = real_data.view(-1, data_dim).to(device)
            real_score = D(real_data)

            d_z_noise = sample_noise(minibatch_size, z_dim, device)
            d_fake_data = G(d_z_noise).detach()
            d_fake_score = D(d_fake_data)

            d_loss, real_loss, fake_loss = standard_d_loss(real_score, d_fake_score)

            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Training generator
            g_z_noise = sample_noise(minibatch_size, z_dim, device)
            fake_data = G(g_z_noise)
            fake_score = D(fake_data)  # this is D(G(z))

            if loss_type == 'ce':
                g_loss = standard_g_loss(fake_score, real_score.detach())
            elif loss_type == 'hack':
                g_loss = heuristic_g_loss(fake_score)

            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()
        
            # Show some progress!
            if batch_idx % progress_update_interval == 0 or batch_idx == 0:
                loss_log = f'epoch #{epoch:<5} batch #{batch_idx:<5}:\n\t' + \
                           f'D_loss: {d_loss:<10.5f}, G_loss: {g_loss:<10.5f}' + \
                           f'\n\tD scores:  real: {torch.mean(real_score):.5f}\t' + \
                           f'fake: {torch.mean(d_fake_score):.5f}\t  G score: {torch.mean(fake_score):.5f}\n'

                print(loss_log)
                if save_model:
                    info_file.write(loss_log)
                    info_file.flush()

                if show_separate_loss:
                    d_real_loss_list.append(real_loss.tolist())
                    d_fake_loss_list.append(fake_loss.tolist())
                else:
                    d_real_loss_list.append(d_loss.tolist())
                #
                g_loss_list.append(g_loss.tolist())

                # plot losses
                update_loss_plot(
                    ax_loss, d_real_loss_list, d_fake_loss_list, g_loss_list,
                    progress_update_interval, show_separate_loss
                )
                
                # plot d(x) and d(g(z))
                d_x_list.append(torch.mean(real_score).item())
                d_g_z_list.append(torch.mean(fake_score).item())
                update_disc_plot(ax_disc, d_x_list, d_g_z_list, progress_update_interval)

                # plot generated data
                update_data_plot(ax_data, fake_data.detach(), epoch, batch_idx)

                # Refresh figure
                fig.canvas.draw()
                fig.canvas.flush_events()

                if save_model and save_figs:
                    f = fig_shots_dir.joinpath(f'shot_{it // progress_update_interval}.png')
                    fig.savefig(f, format='png')
        # End of one epoch

    # End of training
    t2 = time()
    elapsed = round((t2 - t1))
    minutes, seconds = divmod(elapsed, 60)
    elapsed = f'\n\nelapsed time: {minutes:02d}:{seconds:02d}'
    print(elapsed)

    if save_model:
        info_file.write(elapsed)
        info_file.flush()
        info_file.close()
    
    if save_model and save_figs:
        f = fig_shots_dir.joinpath(f'shot_{num_epochs * len(train_loader) // progress_update_interval}.png')
        fig.savefig(f, format='png')

    input('\n\npress Enter to end...\n')


def prepare_plots(info, title=''):
    fig: plt.Figure = plt.figure(1, figsize=(14, 8.0))
    fig.canvas.set_window_title(title)

    ax_data = fig.add_subplot(3, 1, 1)
    ax_loss = fig.add_subplot(3, 1, 2)
    ax_disc = fig.add_subplot(3, 1, 3)

    fig.tight_layout(h_pad=1.55, rect=[0.01, 0.06, 0.99, 0.98])

    ax_data.set_title('Generated images', fontweight='bold')

    ax_loss.annotate(
        info.replace('\n', '  '),
        xy=(0, 0),
        xytext=(2, 40),
        xycoords=('figure pixels', 'figure pixels'),
        textcoords='offset pixels',
        bbox=dict(facecolor='dodgerblue', alpha=0.15),
        size=9.5,
        ha='left'
    )
    ax_loss.set_title('Losses', fontweight='bold')
    ax_loss.grid()

    ax_disc.set_title('Discriminator Outputs', fontweight='bold')
    ax_disc.grid()

    return fig, ax_data, ax_loss, ax_disc


def update_loss_plot(ax: plt.Axes, d_loss, d_fake_loss, g_loss, update_interval, separate=False):
    clear_line(ax, 'd_loss')
    clear_line(ax, 'g_loss')

    x = np.arange(1, len(d_loss) + 1)

    if separate:
        ax.plot(x, np.add(d_loss, d_fake_loss), color='dodgerblue', label='D Loss', gid='d_loss')
        clear_line(ax, 'd_real_loss')
        ax.plot(x, d_loss, color='lightseagreen', label='D Loss(Real)', gid='d_real_loss')
        clear_line(ax, 'd_fake_loss')
        ax.plot(x, d_fake_loss, color='mediumpurple', label='D Loss(Fake)', gid='d_fake_loss')
    else:
        ax.plot(x, d_loss, color='dodgerblue', label='D Loss', gid='d_loss')

    ax.plot(x, g_loss, color='coral', label='G Loss', gid='g_loss', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.0f}' for t in ticks * update_interval])


def update_disc_plot(ax: plt.Axes, d_x, d_g_z, update_interval):
    clear_line(ax, 'dx')
    clear_line(ax, 'dgz')

    x = np.arange(1, len(d_x) + 1)
    ax.plot(x, d_x, color='#308862', label='D(x)', gid='dx')
    ax.plot(x, d_g_z, color='#B23F62', label='D(G(z))', gid='dgz', alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.75)
    ax.set_xlim(left=1, right=len(x) + 0.01)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1, 0.1))
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t:.0f}' for t in ticks * update_interval])


def update_data_plot(ax, fake_data, epoch, batch_idx):
    imgs = fake_data[0:30, :].view(3, 10, 28, 28).cpu().numpy()
    imgs = imgs.swapaxes(1, 2).reshape(3 * 28, 10 * 28)
    ax.clear()
    ax.imshow(imgs, cmap='Greys_r')
    ax.axis('off')

    ax.annotate(
        f'epoch: {epoch}, batch: {batch_idx}',
        xy=(0, 0),
        xytext=(2, 12),
        xycoords=('figure pixels', 'figure pixels'),
        textcoords='offset pixels',
        bbox=dict(facecolor='orange', alpha=0.15),
        size=10,
        ha='left'
    )
    # ax.legend(loc='upper right', framealpha=0.75)




if __name__ == "__main__":
    main(sys.argv)
