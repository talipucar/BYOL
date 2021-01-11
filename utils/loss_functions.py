"""
Library of loss functions.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1

A library of loss functions to be used in Self-Supervised learning setting.
"""

import torch as th

def byol_loss(q, z):
    """
    Loss function: InfoNCE - https://arxiv.org/pdf/2006.07733v1.pdf
    :param q: Predictions of Online network
    :param z: Output of Target network
    :return: loss
    """
    # Get the first dimension of q
    ts, _ = q.size()
    # Compute batch size, which is half of q dim since q is concatenation of qi and qj
    bs = int(ts//2)
    # q is a result of forward pass on online_network using concatenated inputs xi, xj i.e. 2 transformations of input x
    (qi, qj) = th.split(q, bs)
    # z is a result of forward pass on target_network using concatenated inputs xi, xj i.e. 2 transformations of input x
    (zi, zj) = th.split(z, bs)
    # First loss using qi and zj
    loss1 = 2-2*(qi*zj).sum(dim=-1)
    # Second loss using qj and zi
    loss2 = 2-2*(qj*zi).sum(dim=-1)
    # Total loss i.e. symmetric loss
    loss = loss1+loss2
    # Return the mean of symmetric loss
    return loss.mean()

