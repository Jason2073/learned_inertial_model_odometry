import os

import torch

from model import SE3HamNODE
from src.learning.utils.logging import logging


class HNODENetwork:
    def __init__(self, device, model_path, model_name, args):
        self.device = device
        self.model_path = model_path
        self.model_name = model_name
        # get model params and create baseline

        # TODO implement nominal model, as in SE3HamNODE_gt to pretrain to baseline. Also find the udim needed for drone,
        # can we use simply 1d thrust like the prev paper? see notes/questions
        MASS = 0.915 #from blackbird dataset
        c_t_f = 8.004e-3  # meters (c_tau_f coefficient from https://arxiv.org/pdf/1003.2005.pdf section 7)
        d = .13  # dist from center of drone to propeller from blackbird dataset
        m1_nom = torch.linalg.inv(torch.eye(3) * MASS)
        m2_nom = torch.linalg.inv(torch.tensor([[4.9e-2, 0, 0], [0, 4.9e-2, 0], [0, 0, 6.9e-2]])) #intertias from blackbird dataset
        g_nom = torch.tensor([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 1, 1, 1],
                          [0, -d, 0, d],
                          [d, 0, -d, 0],
                          [-c_t_f, c_t_f, -c_t_f, c_t_f]]) #from https://arxiv.org/pdf/1003.2005.pdf eq 1 with added zero rows to fit the shape required by the model
        # we use the SE3HamNode model using a 4 dimensional input with the four propeller thrusts as input
        path = os.path.join(self.model_path, self.model_name + ".pt")
        if os.path.isfile(path):
            logging.info(f"Found model, loading from {path}")
            self.model = SE3HamNODE(device=self.device, udim=4, args=args, pretrain=False)
            # self.model.load_state_dict(torch.load(path))
            self.model.load_state_dict(torch.load(path).get("model_state_dict"))
        else:
            logging.info("No model found, pretraining ...")
            self.model = SE3HamNODE(device=self.device, udim=4, args=args,
                                    m1_nominal=m1_nom, m2_nominal=m2_nom, g_nominal=g_nom)
            # self.model = SE3HamNODE(device=self.device, udim=4, args=args)
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "epoch": 0,
                "args": vars(args),
            }
            torch.save(state_dict, path)
            logging.info(f"Saving pretrained model at {path}")
        self.model.to(self.device)

    def predict(self, x, u):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            u = torch.from_numpy(u).float().to(self.device)
            x = torch.cat((x, u), dim=1)
            dx = self.model(x)
            dx = dx.cpu().numpy()
        return dx

    def get_model(self):
        return self.model
