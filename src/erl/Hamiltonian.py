from model import SE3HamNODE
import torch
import os
from src.learning.utils.logging import logging

class HNODENetwork:
    def __init__(self, device, model_path, model_name, args):
        self.device = device
        self.model_path = model_path
        self.model_name = model_name
        #get model params and create baseline

        #TODO implement nominal model, as in SE3HamNODE_gt to pretrain to baseline. Also find the udim needed for drone,
        # can we use simply 1d thrust like the prev paper? see notes/questions

        # we use the SE3HamNode model using a 4 dimensional input with the four propeller thrusts as input
        path = os.path.join(self.model_path, self.model_name + ".pt")
        if os.path.isfile(path):
            logging.info(f"Found model, loading from {path}")
            self.model = SE3HamNODE(device=self.device, udim=4, args=args, pretrain=False)
            self.model.load_state_dict(torch.load(path))
        else:
            logging.info("No model found, pretraining ...")
            self.model = SE3HamNODE(device=self.device, udim=4, args=args)
            torch.save(self.model.state_dict(), path)
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