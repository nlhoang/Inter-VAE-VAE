import argparse
import datetime
import sys
import random
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from torch import optim
from agent import Agent
from utils import Logger, save_toFile, figure, mh_count_heatmap
eps = 1e-20


def set_seeds(seed):
    if seed == -1:
        seed = random.randint(1, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed: {:.2g}'.format(seed))


def args_define():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1, metavar='N', help='MH(1), No(2), All(3)')
    parser.add_argument('--latent-dim', type=int, default=50, metavar='L', help='latent dimensionality (default: 20)')
    parser.add_argument('--word-length', type=int, default=10, metavar='L', help='word dimensionality (default: 20)')
    parser.add_argument('--dictionary-size', type=int, default=100, metavar='L', help='dictionary size (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--vae-epochs', type=int, default=500, metavar='N', help='No of epochs of gruVAE [default: 100]')
    parser.add_argument('--mh-epochs', type=int, default=500, metavar='N', help='No of epochs of MH naming game [default: 10]')
    parser.add_argument('--dataset', type=str, default='dsprites', help='Datasets [shapes3d, dsprites]')
    parser.add_argument('--D', type=int, default=18432, metavar='N', help='number of data points [32000, 18432]')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR', help='learning rate [default: 1e-3]')
    parser.add_argument('--vae-perception-beta', type=float, default=10.0, metavar='N', help='variational beta [default: 1.0]')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--device', type=str, default='cpu', help='device for training [mps, cuda, cpu]')
    parser.add_argument('--debug', type=bool, default=False, help='debug vs running')
    parser.add_argument('-f', '--file', help='Path for input file')
    return parser.parse_args()


def initialize():
    if args.dataset == 'dsprites':
        args.D = 18432
    elif args.dataset == 'shapes3d':
        args.D = 32000
    else:
        print('Error: Dataset not recognized')
        sys.exit(1)

    if args.debug:
        args.vae1_epochs = 2
        args.vae2_epochs = 2
        args.mh_epochs = 2
        args.eval_sample = 100

    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    sys.stdout = Logger('{}/run.log'.format(runPath))
    print('Expt:', runPath)
    print('RunID:', runId)
    return runPath


def MH_naming_game(A, B, mode=1):
    print('Training MH Naming Game')
    a_optimizer = optim.Adam(A.vae_language.parameters(), lr=args.learning_rate)
    b_optimizer = optim.Adam(B.vae_language.parameters(), lr=args.learning_rate)
    for epoch in range(args.mh_epochs):
        print('====> Epoch: {}'.format(epoch))
        A.train_MH_languageVAE(Speaker=B, optimizer=a_optimizer, mode=mode)
        B.train_MH_languageVAE(Speaker=A, optimizer=b_optimizer, mode=mode)

    A.latents_pos, A.messages = A.vae_language_get_message(A.vae_language, A.dataloader_latent, A.device)
    B.latents_pos, B.messages = B.vae_language_get_message(B.vae_language, B.dataloader_latent, B.device)
    torch.save(A.vae_language.state_dict(), args.run_path + 'a_vae_language.pth')
    torch.save(B.vae_language.state_dict(), args.run_path + 'b_vae_language.pth')
    print('Messages of A:')
    print(A.messages)
    print('Messages of B:')
    print(B.messages)
    print('----------')


def visualization():
    figure(A.acceptedCount, B.acceptedCount, 'Agent A', 'Agent B', args.D, args.run_path + 'acceptFig')
    mh_count_heatmap(A.mh_ratio_count, args.run_path + 'a_mh_ratio')
    mh_count_heatmap(B.mh_ratio_count, args.run_path + 'b_mh_ratio')


if __name__ == "__main__":
    set_seeds(-1)
    args = args_define()
    args.run_path = initialize() + '/'
    print(args)

    mode = args.mode
    if mode == 1:
        print('Communication Protocol: Metropolis-Hastings Naming Game')
    elif mode == 2:
        print('Communication Protocol: All Accepted')
    elif mode == 3:
        print('Communication Protocol: No Communication')
    print('--------------------')

    A = Agent(name='a', args=args)
    B = Agent(name='b', args=args)
    print('Training Agent A - VAE')
    A.train_vae_object()
    print('Training Agent B - VAE')
    B.train_vae_object()

    if mode == 1 or mode == 2 or mode == 3:
        MH_naming_game(A, B, mode)

    save_toFile(path=args.run_path, file_name='a_accept', data_saved=A.acceptedCount, rows=0)
    save_toFile(path=args.run_path, file_name='b_accept', data_saved=B.acceptedCount, rows=0)
    save_toFile(path=args.run_path, file_name='a_mh_count', data_saved=A.mh_ratio_count, rows=1)
    save_toFile(path=args.run_path, file_name='b_mh_count', data_saved=B.mh_ratio_count, rows=1)
    save_toFile(path=args.run_path, file_name='a_messages', data_saved=A.messages, rows=1)
    save_toFile(path=args.run_path, file_name='b_messages', data_saved=B.messages, rows=1)
    visualization()
