from src.erl.utils.utilities import *
from src.learning.utils.logging import logging
def HNODELoss(targ, pred):
    """HNODE loss function.

    Args:
        targ: target values, [px, py, pz, qx, qy, qz, qw, vx, vy, vz]
        pred: predicted values [px,py,pz, R(9x1), vx,vy,vz,...]

    Returns:
        loss: loss value
    """
    loss = 0
    for i in range(targ.shape[2]):
        if (pred[:, 3:12, i] < .01).all():
            logging.info("ZERO PREDICTION!!")
            continue
        norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(pred[:, 3:12, i])
        norm_R = compute_rotation_matrix_from_unnormalized_rotmat(compute_rotation_matrix_from_quaternion(targ[:, 3:7, i]).reshape(-1, 9))
        # R_hat = pred[:, 3:12, i].reshape(-1, 3, 3)
        # R = compute_rotation_matrix_from_quaternion(targ[:, 3:7, i])
        geo_loss, _ = compute_geodesic_loss(norm_R, norm_R_hat)
        loss += geo_loss
        loss += L2_loss(targ[:, 0:3, i], pred[:, 0:3, i])
        loss += L2_loss(targ[:, 7:10, i], pred[:, 12:15, i])
    return loss
