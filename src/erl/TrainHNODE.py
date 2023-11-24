import torch
import numpy as np
import time
import os
from src.learning.utils.logging import logging
from src.erl.data_management.datasets import ModelDataset
from torch.utils.data import DataLoader
from src.learning.utils.argparse_utils import add_bool_arg
import argparse
from src.erl.Hamiltonian import HNODENetwork
from functools import partial
import sys
import json
from src.erl.model.Loss import HNODELoss
from torch.utils.tensorboard import SummaryWriter
import signal

from torchdiffeq import odeint_adjoint as odeint
import numexpr as ne

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
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.add_argument("--continue_from", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=1000, help="max num epochs")
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nonlinearity", type=str, default="relu", help="relu,tanh,gelu")
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

    parser.add_argument("--model_path", type=str, default="pretrain")
    parser.add_argument("--model_name", type=str, default="SE3HamNODE")

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument("--predict_horizon", type=int, default=2)

    add_bool_arg(parser, "save_plots", default=False)
    add_bool_arg(parser, "show_plots", default=False)

    args = parser.parse_args()
    return args

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_error_and_loss(preds, targ):
    loss = HNODELoss(targ, preds)
    err = targ[:, 0:3, :] - preds[:, 0:3, :] #for error just use delta pos for now
    return err, loss

def get_inference(args, model, data_loader, device):
    """
    Get network status
    """
    errors_all, losses_all = [], []

    model.eval()

    for _, (feat, v_init, targ, ts, _, _) in enumerate(data_loader):

        feat = feat.to(device)
        v_init = v_init.to(device)
        targ = targ.to(device)
        ts = ts.to(device)

        #feat has values [poses, rot_mats, vels, w_gyro_calib, full_thrust]
        # for this, we only need the full_thrust, rot_mats, pose, and v_init]
        #model expects [px,py,pz, flatR, vx,vy,vz, wx,wy,wz, 4xthrusts]

        #make this float32
        t_eval = (ts[1,0:2] - ts[1,0]).to(device)
        t_eval = t_eval.type(torch.float32)
        # get network prediction
        y0 = feat[:, :, 0]
        preds = torch.zeros((feat.shape[0], feat.shape[1], feat.shape[2]-1)).to(device)
        for i in range(feat.shape[2]-1):
            dp = odeint(model, y0, t_eval, method=args.solver)
            y0 = dp[1, :, :]
            y0[:, -4:] = feat[:, -4:, i]
            preds[:, :, i] = dp[1, :, :]




        # compute loss
        # errs, loss = get_error_and_loss(dp, targ, learn_configs, device)
        errs, loss = get_error_and_loss(preds, targ)

        # log
        losses_all.append(torch_to_numpy(loss))
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errors_all.append(errs_norm)

    # save
    losses_all = np.asarray(losses_all)
    errors_all = np.concatenate(errors_all, axis=0)

    attr_dict = {
        "errors": errors_all,
        "losses": losses_all
    }

    return attr_dict

def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """
    error = np.mean(attr_dict["errors"])
    loss = np.mean(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_loss_pos/avg", error, epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", loss, epoch)
    logging.info(f"{mode}: average error [m]: {error}")
    logging.info(f"{mode}: average loss: {loss}")

    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1)

def run_train(args, model, train_loader, device, optimizer):
    """
    Train for one epoch
    """

    errors_all, losses_all = [], []
    # torch.autograd.set_detect_anomaly(True)
    for batch, (feat, v_init, targ, ts, _, _) in enumerate(train_loader):
        # logging.info(batch)
        # Here feat is shape (batch_size, 10, prediction_horizon) the feature is created in src/erl/datasets/datasets.py
        # with ModelSequence() class. currently the rows coorespond to [w_gyro_calib, w_thrust, full_thrust]
        #   1. w_gyro_calib is the [wx wy wz] values rotated into the world frame,
        #   2. w_thrust (3 vals) is the mass normalized collective thrust vector rotated into the world frame,
        #   3. full_thrust is the thrusts for each prop, in the body frame in Newtons (not mass normalized atm).

        # v_init is the initial velocity (vx,vy,vz) of the quadrotor in the world frame.

        # targ is shape (batch_size, 10, prediciton_horizon-1) and is the gt state of the drone for time
        # t+1 to t+prediction_horizon. the 10 values are (px, py, pz, qx, qy, qz, qw, vx, vy, vz) in the world frame.

        # vel_diffs = v_init - targ[:, 7:10, 0] should be small, (i checked and it is small, but depends on initial velocity perturbing)

        feat = feat.to(device)
        v_init = v_init.to(device)
        targ = targ.to(device)

        optimizer.zero_grad()

        # get network prediction
        t_eval = (ts[1, 0:2] - ts[1, 0]).to(device)
        t_eval = t_eval.type(torch.float32)
        # get network prediction
        y0 = feat[:, :, 0]
        #TODO, to get the network to work with float32 try setting this to float 32
        preds = torch.zeros((feat.shape[0], feat.shape[1], feat.shape[2] - 1)).to(device)
        for i in range(feat.shape[2] - 1):
            dp = odeint(model, y0, t_eval, method=args.solver)
            y0 = dp[1, :, :].detach().clone()
            y0[:, -4:] = feat[:, -4:, i]
            preds[:, :, i] = dp[1, :, :]

        # compute loss
        errs, loss = get_error_and_loss(preds, targ)

        # log
        losses_all.append(torch_to_numpy(loss))
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errors_all.append(errs_norm)

        # backprop and optimization
        loss.backward()
        optimizer.step()
    # save
    losses_all = np.asarray(losses_all)
    errors_all = np.concatenate(errors_all, axis=0)

    train_dict = {
        "errors": errors_all,
        "losses": losses_all
    }

    return train_dict

def save_model(args, epoch, network, optimizer, interrupt=False):
    if interrupt:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_latest.pt")
    else:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")

def create_necessary_dirs(args):
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.train_list is None:
            raise ValueError("train_list must be specified.")
        if args.dataset is None:
            raise ValueError("dataset must be specified.")
        args.out_dir = os.path.join(args.out_dir)
        if args.out_dir != None:
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints"))
            if not os.path.isdir(os.path.join(args.out_dir, "pretrain")):
                os.makedirs(os.path.join(args.out_dir, "pretrain"))
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints", "model_net")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints", "model_net"))
            if not os.path.isdir(os.path.join(args.out_dir, "logs")):
                os.makedirs(os.path.join(args.out_dir, "logs"))
            with open(
                    os.path.join(args.out_dir, "checkpoints", "model_net", "model_net_parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        if args.val_list is None:
            logging.warning("val_list != specified.")
        if args.continue_from != None:
            if os.path.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
    except ValueError as e:
        logging.error(e)
        return


def prep_wandb(args):
    #TODO implement wandb
    print("wandb not implemented yet!")
    pass

def main():
    args = get_args()
    create_necessary_dirs(args)
    os.environ['NUMEXPR_MAX_THREADS'] = str(args.num_threads)

    torch.set_default_dtype(torch.float32)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    logging.info(f"Using device {device}")

    train_loader, val_loader = None, None
    start_t = time.time()
    train_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.train_list))
    try:
        train_dataset = ModelDataset(
            args.root_dir, args.dataset, train_list, args, args.predict_horizon, mode="train")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
    except OSError as e:
        logging.error(e)
        return
    end_t = time.time()
    logging.info("Loaded train dataset in %.2f s" % (end_t - start_t))
    network = HNODENetwork(device, os.path.join(args.out_dir, args.model_path), args.model_name, args=args)
    model = network.get_model()

    run_validation = False
    val_list = None
    if args.val_list != '':
        run_validation = True
        val_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.val_list))

    #TODO check optimizer / change
    n_params = model.get_num_params()
    params = model.parameters()
    logging.info(f'HNODE network loaded to device {device}')
    logging.info(f"Total number of learning parameters: {n_params}")

    optimizer = torch.optim.Adam(params, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from != None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "inertial_net", "checkpoint_latest.pt")

        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(os.path.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {n_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    attr_dict = get_inference(args, model, train_loader, device)
    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    best_loss = np.mean(attr_dict["losses"])
    # run first validation of the full validation set
    if run_validation:
        try:
            val_dataset = ModelDataset(
                args.root_dir, args.dataset, val_list, args, args.predict_horizon, mode="val")
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(val_dataset)}")

        val_dict = get_inference(args, model, val_loader, device)
        write_summary(summary_writer, val_dict, start_epoch, optimizer, "val")
        best_loss = np.mean(val_dict["losses"])

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_dict = run_train(args, model, train_loader, device, optimizer)
        write_summary(summary_writer, train_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if run_validation:
            # run validation on a random sequence in the validation dataset
            if not args.dataset == 'Blackbird':
                val_sample = np.random.randint(0, len(val_list))
                val_seq = val_list[val_sample]
                logging.info("Running validation on %s" % val_seq)
                try:
                    val_dataset = ModelDataset(
                        args.root_dir, args.dataset, [val_seq], args, args.predict_horizon, mode="val")
                    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
                except OSError as e:
                    logging.error(e)
                    return

            val_attr_dict = get_inference(args, model, val_loader, device)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            current_loss = np.mean(val_attr_dict["losses"])

            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, model, optimizer)
        else:
            attr_dict = get_inference(args, model, train_loader, device)
            current_loss = np.mean(attr_dict["losses"])
            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, model, optimizer)

    logging.info("Training complete.")

if __name__ == "__main__":
  main()