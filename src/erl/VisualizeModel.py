import argparse
import json
import os
import signal
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint_adjoint as odeint

from src.erl.Hamiltonian import HNODENetwork
from src.erl.data_management.datasets import ModelDataset
from src.erl.model.Loss import HNODELoss
from src.learning.utils.argparse_utils import add_bool_arg
from src.learning.utils.logging import logging
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="results/SE3")
    parser.add_argument("--dataset", type=str, default="datasets/Blackbird")
    parser.add_argument("--train_list", type=str, default="train.txt", help="In folder root_dir.")
    # parser.add_argument("--val_list", type=str, default="val.txt", help="In folder root_dir.")
    parser.add_argument("--val_list", type=str, default="", help="In folder root_dir.")
    parser.add_argument("--test_list", type=str, default="test.txt", help="In folder root_dir.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-04)
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.add_argument("--continue_from", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=1000, help="max num epochs")
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nonlinearity", type=str, default="tanh", help="relu,tanh,gelu")
    add_bool_arg(parser, "perturb_orientation", default=True)
    parser.add_argument(
        "--perturb_orientation_theta_range", type=float, default=5.0
    )  # degrees
    parser.add_argument(
        "--perturb_orientation_mean", type=float, default=0.0
    )  # degrees
    parser.add_argument(
        "--perturb_orientation_std", type=float, default=2.0
    )  # degrees
    add_bool_arg(parser, "perturb_bias", default=False)
    parser.add_argument("--gyro_bias_perturbation_range", type=float, default=0.01)
    add_bool_arg(parser, "perturb_init_vel", default=True)
    parser.add_argument("--init_vel_sigma", type=float, default=0.3)

    parser.add_argument("--model_path", type=str, default="pretrain/")
    parser.add_argument("--model_name", type=str, default="SE3HamNODE")

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument("--predict_horizon", type=int, default=5)

    add_bool_arg(parser, "save_plots", default=False)
    add_bool_arg(parser, "show_plots", default=False)

    args = parser.parse_args()
    return args

def visulaize():
    args = get_args()
    torch.set_default_dtype(torch.float32)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info(f"Using device {device}")
    network = HNODENetwork(device, os.path.join(args.out_dir, args.model_path), args.model_name, args=args)
    model = network.get_model()

    n_params = model.get_num_params()
    params = model.parameters()
    logging.info(f'HNODE network loaded to device {device}')
    logging.info(f"Total number of learning parameters: {n_params}")

    #pass an example state
    x = torch.tensor(np.array([[0.0,0.0,0.0]]) , dtype=torch.float32).to(device)
    R = torch.tensor(np.identity(3).reshape((1,9)), dtype=torch.float32).to(device)
    vel = np.array([.1, .1, 0])
    omega = np.array([0, 0 , 0])
    q = torch.hstack([x,R])
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    M_1 = np.linalg.inv(model.M_net1(x).cpu().detach().numpy())
    M_2 = np.linalg.inv(model.M_net2(R).cpu().detach().numpy())
    V_q = model.V_net(q).cpu().detach().numpy()
    g_q = model.g_net(q).cpu().detach().numpy()

    print(f"M1: \n")
    print(M_1)

    print(f"\nM2: \n")
    print(M_2)

    print(f"\nV: \n")
    print(V_q)

    print(f"\ng: \n")
    print(g_q)



if __name__ == "__main__":
    visulaize()