"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/dataloader/dataset_fb.py
"""

from abc import ABC, abstractmethod
import os
import random

import h5py
import numpy as np
from torch.utils.data import Dataset
import src.learning.utils.pose as pose


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass


class ModelSequence(CompiledSequence):
    def __init__(self, bsdir, dtset_fn, seq_fn, **kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.features,
            self.targets,
            self.ts,
            self.gyro_raw,
            self.thrust,
            self.full_thrust,
            self.feat,
            self.traj_target
        ) = (None, None, None, None, None, None, None, None, None)

        data_path = os.path.join(bsdir, dtset_fn, seq_fn)
        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        with h5py.File(os.path.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            thrust = np.copy(f["thrust"])
            i_thrust = np.copy(f["i_thrust"])  # in imu frame
            full_thrust = -1*np.copy(f["full_thrust"])
            traj_target = np.copy(f["traj_target"])

        assert thrust.shape[0] == gyro_calib.shape[0], \
            "Make sure that initial and final times correspond to first and last thrust measurement in %s!" % data_path

        # rotate to world frame
        # w_gyro_calib = np.array([pose.xyzwQuatToMat(T_wi[3:7]) @ w_i for T_wi, w_i in zip(traj_target, gyro_calib)])
        #need to use omega values in the body frame for the HNODE
        imu_to_body_R = np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]]) #rotates from imu frame to body frame
        gyro_calib_body = np.array([w_i for w_i in gyro_calib])
        # w_thrust = np.array([pose.xyzwQuatToMat(T_wi[3:7]) @ t_i for T_wi, t_i in zip(traj_target, i_thrust)])
        rot_mats = np.array([pose.xyzwQuatToMat(T_wi[3:7]) for T_wi in traj_target]).reshape((gyro_calib_body.shape[0], 9))
        poses = np.array([T_wi[0:3] for T_wi in traj_target])


        # rotate velocities into body frame
        # TODO VERIFY! Above they use this matrix to rotate gyro_calib into world frame, so i assume I use the transpose to rotate velocities into body frame
        vels_b = np.array([pose.xyzwQuatToMat(T_wi[3:7]).T @ T_wi[7:] for T_wi in traj_target])
        vels_w = np.array([T_wi[7:] for T_wi in traj_target])

        #set traj_target to use body frame velocities

        new_target = np.zeros((traj_target.shape[0], 15))
        new_target[:, :3] = traj_target[:, :3]
        new_target[:, 3:12] = np.array([pose.xyzwQuatToMat(T_wi[3:7]).flatten() for T_wi in traj_target])
        new_target[:, 12:15] = vels_b
        self.ts = ts
        self.gyro_raw = gyro_raw
        self.thrust = thrust
        self.full_thrust = full_thrust
        # self.feat = np.concatenate([w_gyro_calib, w_thrust, full_thrust, rot_mats], axis=1)
        #TODO find a better way to do the w_ground truth, currently just passing w_gyro_calib,
        # i talked to Sambaran and we're gonna test with this first

        self.feat = np.concatenate([poses, rot_mats, vels_b, gyro_calib_body, full_thrust], axis=1)
        self.traj_target = new_target

    def get_feature(self):
        return self.feat

    def get_target(self):
        return self.traj_target

    # Auxiliary quantities, not used for training.
    def get_aux(self):
        return self.ts, self.gyro_raw, self.thrust, self.full_thrust


class ModelDataset(Dataset):
    def __init__(self, root_dir, dataset_fn, data_list, args, predict_horizon, **kwargs):
        super(ModelDataset, self).__init__()

        # self.sampling_factor = data_window_config["sampling_factor"]
        # self.window_size = int(data_window_config["window_size"])
        # self.window_shift_size = data_window_config["window_shift_size"]
        self.predict_horizon = predict_horizon
        self.g = np.array([0., 0., 9.8082])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
        self.perturb_orientation_mean = args.perturb_orientation_mean
        self.perturb_orientation_std = args.perturb_orientation_std
        self.perturb_bias = args.perturb_bias
        self.gyro_bias_perturbation_range = args.gyro_bias_perturbation_range
        self.perturb_init_vel = args.perturb_init_vel
        self.init_vel_sigma = args.init_vel_sigma
        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "display":
            self.shuffle = False

        # index_map = [[seq_id, index of the last datapoint in the window], ...]
        self.index_map = []
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(root_dir, dataset_fn, data_list[i], **kwargs)
            # feat = np.array([[wx, wy, wz, thrx, thry, thrz], ...])
            # targ = np.array([[x, y, z, qx, qy, qz, qw], ...])
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            N = self.features[i].shape[0]
            #index map with i for each datalist, and j cooresponds for each value in the datalist
            self.index_map += [ [i, j] for j in range(0, N-int(predict_horizon), 1)]

            times, raw_gyro_meas, thrusts, full_thrusts = seq.get_aux()
            self.ts.append(times)

            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.thrusts.append(thrusts)
                
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        idxs = frame_id
        idxe = frame_id + self.predict_horizon
        indices = range(idxs, idxe, 1)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]
        feat[1:,:-4] = 0 #zero out values of the feature that we cannot use.

        # vel
        if idxs <= 0:
            # t0m1 = self.ts[seq_id][idxs]
            # pw0m1 = self.targets[seq_id][idxs][0:3]
            # print("idxs is <= 0")
            vw0 = self.targets[seq_id][idxs][12:15]
        else:
            # t0m1 = self.ts[seq_id][idxs-1]
            # pw0m1 = self.targets[seq_id][idxs-1][0:3]
            vw0 = self.targets[seq_id][idxs-1][12:15]

        # t0p1 = self.ts[seq_id][idxs+1]
        # pw0p1 = self.targets[seq_id][idxs+1][0:3]
        # vw0 = (pw0p1 - pw0m1) / (t0p1 - t0m1)

        # pos
        # pw0 = self.targets[seq_id][idxs][0:3]
        # pw1 = self.targets[seq_id][idxe][0:3]

        # targ = pw1 - pw0
        targ = self.targets[seq_id][idxs + 1:idxe + 1 + (1 if self.predict_horizon == 1 else 0), :].T

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        thrust_i = np.zeros((3,))
        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            thrust_i = self.thrusts[seq_id][indices]
            
        if self.mode == "train":
            # perturb biases
            # if self.perturb_bias:
            #     random_bias = [
            #         (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
            #         (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
            #         (random.random() - 0.5) * self.gyro_bias_perturbation_range / 0.5,
            #     ]
            #     feat[:, 0] = feat[:, 0] + random_bias[0]
            #     feat[:, 1] = feat[:, 1] + random_bias[1]
            #     feat[:, 2] = feat[:, 2] + random_bias[2]

            if self.perturb_orientation:
                vec_rand = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
                vec_rand = vec_rand / np.linalg.norm(vec_rand)

                theta_rand = (
                        random.random() * np.pi * self.perturb_orientation_theta_range / 180.0)
                # theta_deg = np.random.normal(self.perturb_orientation_mean, self.perturb_orientation_std)
                # theta_rand = theta_deg * np.pi / 180.0

                R_mat = pose.fromAngleAxisToRotMat(theta_rand, vec_rand)
                #
                # feat[:, 0:3] = np.matmul(R_mat, feat[:, 0:3].T).T
                rotated = np.matmul(R_mat, feat[0, 3:12].reshape((3,3)).T).T #perterb the orientation
                feat[0, 3:12] = rotated.reshape((9,))

            # perturb initial velocity
            if self.perturb_init_vel:
                dv = np.array([
                    np.random.normal(scale=self.init_vel_sigma),
                    np.random.normal(scale=self.init_vel_sigma),
                    np.random.normal(scale=2*self.init_vel_sigma)])
                vw0 = vw0 + dv
                feat[0, 12:15] = feat[0, 12:15] + dv #TODO check this

        return feat.astype(np.float32).T, vw0.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, thrust_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

