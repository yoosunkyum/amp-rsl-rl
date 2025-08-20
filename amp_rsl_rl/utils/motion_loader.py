# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

from isaaclab.utils.math import quat_apply_inverse


def rotation_to_local_angular_velocity(rot_seq: Rotation, dt: float) -> np.ndarray:
    """
    Compute local angular velocity from scipy Rotation sequence.

    Args:
        rot_seq: scipy Rotation object of length N (e.g., from R.from_matrix or R.from_quat)
        dt: time step between each rotation

    Returns:
        omega_local: (N-1, 3) ndarray of angular velocity vectors in local frame
    """
    # R_rel = R_t.inv() * R_{t+1}, each in local frame
    R_t = rot_seq[:-1]
    R_t1 = rot_seq[1:]
    # print("R_t : ",R_t.as_quat()[:5])
    # print("R_t1 : ",R_t1.as_quat()[:5])
    # Relative rotations from local frame
    R_rel = R_t.inv() * R_t1  # shape (N-1,)
    # print("R_rel : ", R_rel.as_quat()[:5])

    # Rotation vector (axis * angle) → angular velocity
    rotvec = R_rel.as_rotvec()  # shape (N-1, 3)
    # print("rotvec : ",rotvec[:5])
    omega_local = rotvec / dt
    return omega_local




def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())
        dataset_names.append(file.replace(".npy", ""))

    return dataset_names


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: (currently zeros)
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: (currently zeros)
        - base_quat: orientation quaternion as torch.Tensor in wxyz order

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted internally to `wxyz` format to be compatible with IsaacLab conventions.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_pos_z: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    # projected_gravity: Union[Rotation, torch.Tensor]
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_pos_z, np.ndarray):
            self.base_pos_z = to_tensor(self.base_pos_z)
            
        if isinstance(self.base_quat, Rotation):
        # if isinstance(self.projected_gravity, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            # convert to wxyz
            quat_wxyz = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]],
                device=self.device,
                dtype=torch.float32,
            )
            self.base_quat = quat_wxyz
        
        
        gravity = torch.tensor([0.0, 0.0, -1.0],device=self.device).expand(quat_wxyz.shape[:-1] + (3,))
        self.projected_gravity = quat_apply_inverse(self.base_quat, gravity)
    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Concatenated observation tensor
        """
        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
                self.base_pos_z[indices],##only height information is needed
                # self.base_quat[indices]
                self.projected_gravity[indices]
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    Responsibilities:
      - Loading .npy files containing motion data
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        Each .npy contains a dict with keys:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the .npy motion files
        dataset_names: List of dataset filenames (no extension)
        dataset_weights: List of sampling weights (for minibatch sampling)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
        expected_joint_names: (Optional) override for joint ordering
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # ─── Build union of all joint names if not provided ───
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                p = dataset_path_root / f"{name}.npy"
                info = np.load(str(p), allow_pickle=True).item()
                for j in info["joints_list"]:
                    if j not in seen:
                        seen.add(j)
                        joint_union.append(j)
            expected_joint_names = joint_union
        # ─────────────────────────────────────────────────────────
        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            md = self.load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def _resample_data_Rn(
        self,
        data: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> MotionData:
        """
        Loads and processes one motion dataset.

        Returns:
            MotionData instance
        """
        data = np.load(str(dataset_path), allow_pickle=True).item()
        dataset_joint_names = data["joints_list"]

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )
        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        resampled_base_ang_vel_local = rotation_to_local_angular_velocity(resampled_base_orientations, simulation_dt)
        resampled_base_ang_vel_local = np.vstack([resampled_base_ang_vel_local, resampled_base_ang_vel_local[-1:]])

        zeros = np.zeros_like(resampled_base_lin_vel_mixed)

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=zeros,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            # base_ang_velocities_local=zeros,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_pos_z=resampled_base_positions[:,2:3],
            base_quat=resampled_base_orientations,
            # projected_gravity=resampled_base_orientations,
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)


import numpy as np
import os
import torch
from typing import Optional


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, 
                 device: torch.device,
                 dataset_path_root: Path ,
                 dataset_names : List[str],
                 dataset_weights: List[float],
                 simulation_dt: float,
                 slow_down_factor: int,
                 expected_joint_names: Union[List[str], None] = None,
                #  motion_file: str, 
                 ) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        # if isinstance(dataset_path_root, str):
        #     dataset_path_root = Path(dataset_path_root)
        for name in dataset_names:
            # motion_file = dataset_path_root / f"{name}.npz"
            motion_file = "C:/Research/isaaclab_amp_rsl_rl/motions/g1/g1_walk.npz"
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.sim_dt = simulation_dt
        self.duration = self.dt * (self.num_frames - 1)

        self.motion_dof_indexes = self.get_dof_index(expected_joint_names)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert (
            duration <= self.duration
        ), f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions[:,self.motion_dof_indexes], blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities[:,self.motion_dof_indexes], blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )
        
    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int 
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:    
        for _ in range(num_mini_batch):
            num_samples = mini_batch_size
            current_times = self.sample_times(num_samples)
            times = (
                np.expand_dims(current_times, axis=-1)
                # - self.dt * np.arange(0, 2)
                - self.sim_dt * np.arange(0, 2)
            ).flatten()

            # print(f"times : {times}")
            #get motions
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = self.sample(num_samples, times)
            # print(f"dof_positions size : {dof_positions.size()}")
            
            #refine motion dataset
            base_pos_z = body_positions[:,0,2:3]
            
            gravity = torch.tensor([0.0, 0.0, -1.0],device=self.device).expand(body_rotations[:,0,:].shape[:-1] + (3,))
            projected_gravity = quat_apply_inverse(body_rotations[:,0,:], gravity)
            
            base_lin_velocities_local = body_linear_velocities[:,0,:]
            base_ang_velocities_local = body_angular_velocities[:,0,:]
            
            #concatenate & divide
            # obs = torch.cat(
            #     (
            #         dof_positions,
            #         dof_velocities,
            #         base_lin_velocities_local,
            #         base_ang_velocities_local,
            #         base_pos_z,
            #         projected_gravity,
            #     ),
            #     dim=-1                            
            # )
            obs = torch.cat(
                (
                    dof_positions,
                    dof_velocities,
                    base_lin_velocities_local,
                    base_ang_velocities_local,
                    base_pos_z,
                    projected_gravity,
                ),
                dim=-1                            
            )
            # print(f"obs size : {obs.size()}")
            idx = np.arange(0,mini_batch_size)
            #yield current & next motion samples
            yield obs[1+2*idx,:], obs[2*idx,:]
    
    def selected_generator(
        self, times: np.ndarray
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:    
        num_samples = 1

        # print(f"times : {times}")
        #get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self.sample(num_samples, times)
        # print(f"dof_positions size : {dof_positions.size()}")
        
        #refine motion dataset
        base_pos_z = body_positions[:,0,2:3]
        
        gravity = torch.tensor([0.0, 0.0, -1.0],device=self.device).expand(body_rotations[:,0,:].shape[:-1] + (3,))
        projected_gravity = quat_apply_inverse(body_rotations[:,0,:], gravity)
        
        base_lin_velocities_local = body_linear_velocities[:,0,:]
        base_ang_velocities_local = body_angular_velocities[:,0,:]
        
        #concatenate & divide
        obs = torch.cat(
            (
                dof_positions,
                dof_velocities,
                base_lin_velocities_local,
                base_ang_velocities_local,
                base_pos_z,
                projected_gravity,
            ),
            dim=-1                            
        )

        return obs

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes