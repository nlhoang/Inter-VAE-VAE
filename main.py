import argparse
import datetime
import sys
from pathlib import Path
from tempfile import mkdtemp
import torch
from torch import optim
from agent import Agent
from utils import Logger, save_toFile, figure, set_seeds


def args_define():
    parser = argparse.ArgumentParser(description='Train Inter-VAE+VAE models.')
    parser.add_argument('--mode', type=int, default=1, metavar='N', help='MH(1), No(2), All(3)')
    parser.add_argument('--dataset', type=str, default='dsprites', help='Datasets [shapes3d, dsprites]')
    parser.add_argument('--langCoder', type=str, default='TfmEnc', help='Language Coder [TfmEnc, TfmDec, LSTM, GRU]')
    parser.add_argument('--latent-dim', type=int, default=50, metavar='L', help='latent dimensionality (default: 20)')
    parser.add_argument('--word-length', type=int, default=20, metavar='L', help='word dimensionality (default: 20)')
    parser.add_argument('--dictionary-size', type=int, default=100, metavar='L', help='dictionary size (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--mutual-epochs', type=int, default=10, metavar='N', help='No of epochs of mutual [default: 100]')
    parser.add_argument('--vae-epochs', type=int, default=50, metavar='N', help='No of epochs of VAE perception [default: 100]')
    parser.add_argument('--mh-epochs', type=int, default=100, metavar='N', help='No of epochs of MH naming game [default: 100]')
    parser.add_argument('--D', type=int, default=0, metavar='N', help='number of data points [32000, 18432]')
    parser.add_argument('--learning-rate', type=float, default=1e-5, metavar='LR', help='learning rate [default: 1e-3]')
    parser.add_argument('--vae-perception-beta', type=float, default=5.0, metavar='N', help='variational beta [default: 1.0]')
    parser.add_argument('--vae-language-beta', type=float, default=1.0, metavar='N', help='variational beta [default: 1.0]')
    parser.add_argument('--run-path', type=str, default=None, help='directory for saving models')
    parser.add_argument('--device', type=str, default='mps', help='device for training [mps, cuda, cpu]')
    parser.add_argument('--debug', type=bool, default=True, help='debug vs running')
    parser.add_argument('-f', '--file', help='Path for input file')
    return parser.parse_args()


def initialize():
    if args.dataset == 'dsprites':
        args.D = 18432
        args.word_length = 10
    else:  # args.dataset == 'shapes3d':
        args.D = 32000
        args.word_length = 20

    if args.debug:
        args.mutual_epochs = 2
        args.vae_epochs = 2
        args.mh_epochs = 2

    runId = args.dataset + '-' + args.langCoder + '-' + datetime.datetime.now().isoformat()
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


if __name__ == "__main__":
    args = args_define()
    args.run_path = initialize() + '/'
    print(args)
    set_seeds(-1)

    if args.mode == 1:
        print('Communication Protocol: Metropolis-Hastings Naming Game')
    elif args.mode == 2:
        print('Communication Protocol: No Communication')
    else:  # args.mode == 3:
        print('Communication Protocol: All Accepted')
    print('--------------------')

    A = Agent(name='a', args=args)
    B = Agent(name='b', args=args)

    for i in range(args.mutual_epochs):
        print(f' ***** Mutual Epoch [{i+1}/{args.mutual_epochs}] *****')
        print('Training Agent A - Perception VAE')
        A.train_vae_perception()
        print('Training Agent B - Perception VAE')
        B.train_vae_perception()
        MH_naming_game(A, B, args.mode)
        print('--------------------')

    A.vae_language_get_message()
    B.vae_language_get_message()
    torch.save(A.vae_perception.state_dict(), args.run_path + 'a_vae_perception.pth')
    torch.save(A.vae_language.state_dict(), args.run_path + 'a_vae_language.pth')
    torch.save(B.vae_perception.state_dict(), args.run_path + 'b_vae_perception.pth')
    torch.save(B.vae_language.state_dict(), args.run_path + 'b_vae_language.pth')
    print('Messages of A:')
    print(A.messages)
    print('Messages of B:')
    print(B.messages)
    print('------------------')

    save_toFile(path=args.run_path, file_name='a_accept', data_saved=A.acceptedCount, rows=0)
    save_toFile(path=args.run_path, file_name='b_accept', data_saved=B.acceptedCount, rows=0)
    save_toFile(path=args.run_path, file_name='a_mh_count', data_saved=A.mh_ratio_count, rows=1)
    save_toFile(path=args.run_path, file_name='b_mh_count', data_saved=B.mh_ratio_count, rows=1)
    save_toFile(path=args.run_path, file_name='a_messages', data_saved=A.messages, rows=1)
    save_toFile(path=args.run_path, file_name='b_messages', data_saved=B.messages, rows=1)
    figure(A.acceptedCount, B.acceptedCount, 'Agent A', 'Agent B', args.D, args.run_path + 'acceptFig')
