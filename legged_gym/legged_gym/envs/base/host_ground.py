from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

from legged_gym.envs.g1.g1_utils import (
    MotionLib, 
    load_imitation_dataset,
    compute_residual_observations,
    tolerance
)
from legged_gym.utils.math import (
    quat_rotate,
    euler_xyz_to_quat,
    quat_apply_yaw,
    quat_apply_yaw_inverse,
    quat_mul_yaw_inverse,
    quat_conjugate,
    quat_apply,
    quat_mul,
    quat_to_euler_xyz,
    quat_to_rot6d,
    quat_to_angle_axis,
    torch_rand_float,
    quat_conjugate,
)


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.num_real_dofs = cfg.env.num_dofs

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.num_one_step_obs = self.cfg.env.num_one_step_observations #if not self.cfg.env.add_force else self.cfg.env.num_one_step_observations + 1
        self.actor_history_length = self.cfg.env.num_actor_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.unactuated_time = self.cfg.env.unactuated_timesteps
        self.unactuated_time *= 0.02 / self.dt
        self.is_gaussian = cfg.rewards.is_gaussian

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.actions *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time 
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            # vertical pulling force
            if self.cfg.curriculum.pull_force:
                force_tensor = torch.zeros([self.num_envs, self.num_bodies, 3], device=self.device)
                force_tensor[:, self.base_indices, 2] = self.force 

                force_tensor *= (self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time).unsqueeze(1)
                if not self.cfg.curriculum.no_orientation:
                    force_tensor *= (self.projected_gravity[:, 2] < -0.8).unsqueeze(1).unsqueeze(1)
                force_tensor = gymtorch.unwrap_tensor(force_tensor)
                self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.real_episode_length_buf += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.init_rpy = None
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_dof_pos[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length 
        self.reset_buf |= self.time_out_buf

        self.dof_vel_out = (torch.abs(self.dof_vel.max(dim=1).values) > self.cfg.curriculum.dof_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.dof_vel_out

        self.base_vel_out = (torch.norm(self.base_lin_vel[:, :3], dim=-1) > self.cfg.curriculum.base_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.base_vel_out

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
            
        self.extras["episode"] = {}
        self.extras['episode']['base_height'] = self.old_headheight[env_ids].mean()

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.update_force_curriculum(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_last_dof_pos[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.real_episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.old_headheight[env_ids] = 0
        self.max_headheight[env_ids] = 0
        self.feet_ori[env_ids] = 0
        # fill extras
        self.delay_buffer[:, env_ids, :] = 0.
        
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
        # self._reset_motions(env_ids)
        self.extras['episode']["force"] = self.force.mean()
        self.extras['episode']['action_scale'] = self.action_rescale

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx[env_ids] = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(len(env_ids), ), device=self.device)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        if not self.is_gaussian:
            raise NotImplementedError
        else:
            self.rew_buf[:, :] = 0
            task_group_index = self.reward_groups.index('task')
            self.rew_buf[:, task_group_index] = 1
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]() * self.reward_scales[name]
                if len(rew.shape) == 2 and rew.shape[1] == 1:
                    rew = rew.squeeze(1)
                self.rew_buf[:, task_group_index] *= rew # follow "Learning to Get Up"
                self.episode_sums[name] += rew

        for i in range(len(self.constraints)):
            name = self.constraint_names[i]
            reward_group_name = name.split('_')[0]
            rew = self.constraints[i]() * self.constraints_scales[name]
            task_group_index = self.reward_groups.index(reward_group_name)

            self.rew_buf[:, task_group_index] += rew
            self.episode_sums[name] += rew
            if self.cfg.constraints.only_positive_rewards:
                self.rew_buf[:, task_group_index] = torch.clip(self.rew_buf[:, task_group_index], min=0.)

        if "termination" in self.constraints_scales:
            rew = self._reward_termination() * self.constraints_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        for rg in self.reward_groups:
            idx = self.reward_groups.index(rg)
            self.episode_sums[rg] = self.rew_buf[:, idx]

    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat(( 
                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                self.projected_gravity,
                                self.dof_pos * self.obs_scales.dof_pos,
                                self.dof_vel * self.obs_scales.dof_vel,
                                self.actions,
                                self.action_rescale + (torch.rand_like(self.action_rescale) - 0.5) * 0.05,
                                ),dim=-1)
        
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec

        current_obs *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_obs), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
    
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # hard limits
                self.dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_link_index].mass = self.default_rigid_body_mass[self.torso_link_index] + self.payload[env_id, 0]

        if self.cfg.domain_rand.randomize_com_displacement:
            props[self.torso_link_index].com = self.default_com_torso + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(0, len(props)):
                if i == self.torso_link_index:
                    pass
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.action_rescale
    
        self.joint_pos_target = self.dof_pos + actions_scaled
        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concat((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            self.joint_pos_target = self.dof_pos + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            self.joint_pos_target = self.dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains *  self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques = self.motor_strength *  torques + self.actuation_offset
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device) + int(self.cfg.domain_rand.random_pose) * torch_rand_float(-1, 1, (len(env_ids), self.num_dof), device=self.device) 
            self.dof_vel[env_ids] = 0.

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_force_curriculum(self, env_ids):
        if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
            self.force[env_ids] = (self.force[env_ids] - 20).clamp(0, np.inf)
            self.action_rescale[env_ids] = (self.action_rescale[env_ids] - 0.02).clamp(0.25, np.inf)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        start_index = 6
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)# if not self.cfg.env.add_force else torch.zeros(start_index + 3 * self.num_actions + 1, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level

        noise_vec[start_index:start_index+self.num_real_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[start_index+self.num_real_dofs:start_index+2*self.num_real_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[start_index+2*self.num_real_dofs:start_index+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_states = self.dof_state.view(self.num_envs, self.num_dof, 2)
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_last_dof_pos = torch.zeros_like(self.dof_pos)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.old_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.max_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.feet_ori = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.force = self.cfg.curriculum.force * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.action_rescale = self.cfg.control.action_scale * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            print("Name of joint is: ", name)
            angle = self.cfg.init_state.default_joint_angles[name]
            print("Angle for each joint is: ", angle)
            self.default_dof_pos[i] = angle
            self.target_dof_pos[:, i] = self.cfg.init_state.target_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name] 
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.height_noise_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2
        if self.cfg.domain_rand.delay:
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= 1 #self.dt
        
        for key in list(self.constraints_scales.keys()):
            scale = self.constraints_scales[key]
            if scale==0:
                self.constraints_scales.pop(key) 
            else:
                self.constraints_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.reward_functions.append(getattr(self, name))

        self.constraints = []
        self.constraint_names = []
        for name, scale in self.constraints_scales.items():
            # import ipdb; ipdb.set_trace()
            self.constraint_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.constraints.append(getattr(self, name))        

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        for name in self.constraint_names:
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s and 'auxiliary' not in s]
        penalized_contact_names = []
        # import ipdb; ipdb.set_trace()
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.torso_link_index = body_names.index("waist_yaw_link")

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            if i == 0:
                # self.default_com = copy.deepcopy(body_props[0].com)
                self.default_com_torso = copy.deepcopy(body_props[self.torso_link_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
            
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        left_shoulder_names = [s for s in body_names if self.cfg.asset.left_shoulder_name in s and 'keyframe' not in s]
        right_shoulder_names = [s for s in body_names if self.cfg.asset.right_shoulder_name in s and 'keyframe' not in s]
        self.left_shoulder_indices = torch.zeros(len(left_shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_shoulder_names)):
            self.left_shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_shoulder_names[i])
        self.right_shoulder_indices = torch.zeros(len(right_shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_shoulder_names)):
            self.right_shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_shoulder_names[i])

        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])

        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        base_name = [s for s in body_names if self.cfg.asset.base_name in s]
        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_name)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name[i])

        # import ipdb; ipdb.set_trace()
        left_knee_names = [s for s in body_names if self.cfg.asset.left_knee_name in s and 'keyframe' not in s]
        right_knee_names = [s for s in body_names if self.cfg.asset.right_knee_name in s and 'keyframe' not in s]
        self.left_knee_indices = torch.zeros(len(left_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_knee_names)):
            self.left_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_knee_names[i])
        self.right_knee_indices = torch.zeros(len(right_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_knee_names)):
            self.right_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_knee_names[i])

        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])

        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        # import ipdb; ipdb.set_trace()
        self.head_indices = torch.zeros(len(self.head_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.head_names):
            self.head_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        self.left_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_roll_joints)):
            self.left_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_roll_joints[i])
            
        self.right_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_roll_joints)):
            self.right_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_roll_joints[i])
            
        self.hip_roll_joint_indices = torch.cat((self.left_hip_roll_joint_indices, self.right_hip_roll_joint_indices))

        self.left_knee_joint_indices = torch.zeros(len(self.cfg.asset.left_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_knee_joints)):
            self.left_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_knee_joints[i])
            
        self.right_knee_joint_indices = torch.zeros(len(self.cfg.asset.right_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_knee_joints)):
            self.right_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_knee_joints[i])


        self.left_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_pitch_joints)):
            self.left_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_pitch_joints[i])
            
        self.right_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_pitch_joints)):
            self.right_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_pitch_joints[i])
            
        self.hip_pitch_joint_indices = torch.cat((self.left_hip_pitch_joint_indices, self.right_hip_pitch_joint_indices))

        self.all_hip_joint_indices = torch.cat([self.hip_pitch_joint_indices, self.hip_roll_joint_indices, self.hip_joint_indices])

        self.left_shoulder_roll_joint_indices = torch.zeros(len(self.cfg.asset.left_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_shoulder_roll_joints)):
            self.left_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_shoulder_roll_joints[i])
            
        self.right_shoulder_roll_joint_indices = torch.zeros(len(self.cfg.asset.right_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_shoulder_roll_joints)):
            self.right_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_shoulder_roll_joints[i])

        self.shoulder_roll_joint_indices = torch.cat((self.left_shoulder_roll_joint_indices, self.right_shoulder_roll_joint_indices))

        self.left_arm_joint_indices = torch.zeros(len(self.cfg.asset.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.asset.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_arm_joints[i])

        # import ipdb; ipdb.set_trace()
        self.upper_body_joint_indices = torch.cat([self.right_arm_joint_indices, self.left_arm_joint_indices, self.waist_joint_indices])
        self.lower_body_joint_indices = torch.cat([self.all_hip_joint_indices, self.knee_joint_indices, self.ankle_joint_indices])

        left_upper_body_names = []
        for target_name in self.cfg.asset.left_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    left_upper_body_names.append(source_name)
        self.left_upper_body_indices = torch.zeros(len(left_upper_body_names), dtype=torch.long, device=self.device)
        self.left_upper_body_names = left_upper_body_names
        for i, name in enumerate(left_upper_body_names):
            self.left_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
        
        right_upper_body_names = []
        for target_name in self.cfg.asset.right_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    right_upper_body_names.append(source_name)
        self.right_upper_body_indices = torch.zeros(len(right_upper_body_names), dtype=torch.long, device=self.device)
        self.right_upper_body_names = right_upper_body_names
        for i, name in enumerate(right_upper_body_names):
            self.right_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_lower_body_names = []
        for target_name in self.cfg.asset.left_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    left_lower_body_names.append(source_name)
        self.left_lower_body_indices = torch.zeros(len(left_lower_body_names), dtype=torch.long, device=self.device)
        self.left_lower_body_names = left_lower_body_names
        for i, name in enumerate(left_lower_body_names):
            self.left_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_lower_body_names = []
        for target_name in self.cfg.asset.right_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    right_lower_body_names.append(source_name)
        self.right_lower_body_indices = torch.zeros(len(right_lower_body_names), dtype=torch.long, device=self.device)
        self.right_lower_body_names = right_lower_body_names
        for i, name in enumerate(right_lower_body_names):
            self.right_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_ankle_names = []
        for target_name in self.cfg.asset.left_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    left_ankle_names.append(source_name)
        self.left_ankle_indices = torch.zeros(len(left_ankle_names), dtype=torch.long, device=self.device)
        self.left_ankle_names = left_ankle_names
        for i, name in enumerate(left_ankle_names):
            self.left_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_ankle_names = []
        for target_name in self.cfg.asset.right_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    right_ankle_names.append(source_name)
        self.right_ankle_indices = torch.zeros(len(right_ankle_names), dtype=torch.long, device=self.device)
        self.right_ankle_names = right_ankle_names
        for i, name in enumerate(right_ankle_names):
            self.right_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.constraints_scales = class_to_dict(self.cfg.constraints.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws target body position and orientation
        """
        self.gym.clear_lines(self.viewer)
        self._refresh_tensor_state()
        terrain_sphere = gymutil.WireframeSphereGeometry(0.02, 5, 5, None, color=(1, 1, 0))
        marker_sphere = gymutil.WireframeSphereGeometry(0.03, 5, 5, None, color=(0, 1, 1))
        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            base_pos = self.base_pos[i].cpu().numpy()

            motion_body_pos = self.motion_dict["keyframe_pos"][i].clone()
            motion_body_pos = motion_body_pos.cpu().numpy()
            motion_body_quat = self.motion_dict["keyframe_quat"][i].cpu().numpy()

            for j in range(len(self.keyframe_indices)):
                x, y, z = motion_body_pos[j, 0], motion_body_pos[j, 1], motion_body_pos[j, 2]
                a, b, c, d = motion_body_quat[j, 0], motion_body_quat[j, 1], motion_body_quat[j, 2], motion_body_quat[j, 3]
                target_sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(a, b, c, d))
                gymutil.draw_lines(marker_sphere, self.gym, self.viewer, self.envs[i], target_sphere_pose)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_sphere_pose)

    def _refresh_tensor_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
    #-----------------------------task rewards-----------------------------
    def _reward_orientation(self):
        if not self.is_gaussian:
            mse_error = torch.sum(torch.square(self.projected_gravity - torch.tensor([0, 0 ,1], device=self.device)), dim=-1)
            reward = torch.exp(mse_error/self.cfg.rewards.orientation_sigma) * (self.root_states[:, 2] > 0.4)
        else:
            base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
            reward = tolerance(-self.projected_gravity[:, 2], [self.cfg.rewards.orientation_threshold, np.inf], 1., 0.05) #-1 
        return reward

    def _reward_head_height(self):
        if not self.is_gaussian:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            return head_height.squeeze(1).clamp(0, 1)
        else:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            feet_height = self.rigid_body_states[:, self.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
            head_height -= feet_height
            reward = tolerance(head_height, (self.cfg.rewards.target_head_height, np.inf), self.cfg.rewards.target_head_margin, 0.1)
            delta_max_headheight = head_height - self.max_headheight
            delta_headheight = head_height - self.old_headheight
            self.max_headheight = torch.max(torch.cat((head_height, self.old_headheight), dim=1), dim=1)[0].unsqueeze(-1)
            self.old_headheight = head_height
            return reward


    #-----------------------------regularization rewards-----------------------------
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) 

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    #-----------------------------style rewards-----------------------------
    def _reward_waist_deviation(self):
        wrist_dof = self.dof_pos[:, self.waist_joint_indices]
        reward = (torch.abs(wrist_dof) > 0.8).float()
        return reward.squeeze(1)

    def _reward_hip_yaw_deviation(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 1.0) | (torch.min(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 0.5)
        return reward

    def _reward_hip_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] >  1.0) | (torch.min(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 0.5)
        return reward

    def _reward_shoulder_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.shoulder_roll_joint_indices]
        reward = (self.dof_pos[:, self.shoulder_roll_joint_indices[0]] < -0.02) | (self.dof_pos[:, self.shoulder_roll_joint_indices[1]] > 0.02)
        return reward

    def _reward_left_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        left_foot_xy = self.rigid_body_states[:, self.left_foot_indices, :2].squeeze(1)
        mse_error = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1).clamp(0.3, np.inf)
        reward = torch.exp(mse_error * self.cfg.rewards.left_foot_displacement_sigma) *  (self.rigid_body_states[:, self.left_foot_indices, 2] < 0.3).squeeze(1)

        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup

    def _reward_right_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        right_foot_xy = self.rigid_body_states[:, self.right_foot_indices, :2].squeeze(1)
        mse_error = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1).clamp(0.3, np.inf)
        reward = torch.exp(mse_error * self.cfg.rewards.right_foot_displacement_sigma) * (self.rigid_body_states[:, self.right_foot_indices, 2] < 0.3).squeeze(1)

        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return reward * standup

    def _reward_knee_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.knee_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.knee_joint_indices]), dim=-1)[0] > 2.4) | (torch.min(self.dof_pos[:, self.knee_joint_indices], dim=-1)[0] < 0.1)
        return reward

    def _reward_shank_orientation(self):
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()

        left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(left_knee_pos - left_foot_pos, dim=-1)
        right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(right_knee_pos - right_foot_pos, dim=-1)

        feet_orientation = torch.mean(torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1)

        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1) * base_height#.unsqueeze(1) 

        if self.cfg.constraints.post_task:
            standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward 

    def _reward_ground_parallel(self):
        left_ankle_pos = self.rigid_body_states[:, self.left_ankle_indices, 2].clone() * 10
        right_ankle_pos = self.rigid_body_states[:, self.right_ankle_indices, 2].clone() * 10
        var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
        var = torch.mean(torch.concat([left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1), dim=-1)
        reward = var < 0.03

        if self.cfg.constraints.post_task:
            standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        return reward

    def _reward_feet_distance(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
        reward = tolerance(feet_distances, [0, 0.4], 0.38, 0.05)
        return (feet_distances > 0.9).squeeze(1)

    def _reward_style_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
 
    #--------------------------post-task rewards-----------------------------
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
 
    def _reward_lin_vel_xy(self):
        # Penalize z axis base linear velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * -5) * base_height

    def _reward_feet_height_var(self):
        left_foot_height = self.rigid_body_states[:, self.left_foot_indices, 2].clone() * 10
        right_foot_height = self.rigid_body_states[:, self.right_foot_indices, 2].clone() * 10
        feet_distance = torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(feet_distance * -2) * standup
    
    def _reward_target_upper_dof_pos(self):
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.target_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma) 
        reward = reward * standup
        return reward
    
    def _reward_target_orientation(self):
        # Penalize non flat base orientation
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -5) * standup

    def _reward_target_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        return torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * - 20) * standup