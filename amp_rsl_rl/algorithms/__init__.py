# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents using AMP."""

from .amp_ppo import AMP_PPO
from .add_ppo import ADD_PPO

__all__ = ["AMP_PPO", "ADD_PPO"]
