# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 3. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import isaacgym
import numpy as np
import mujoco
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import CAS02CfgGround19
from legged_gym.utils.mj_logger import MjLogger
import torch

from datetime import datetime
import time
import csv
# import pygame
from threading import Thread


# python legged_gym/legged_gym/scripts/mujoco_scripts/sim2sim_19dof.py --load_model /home/casbot/ZHZ_ws/HoST/legged_gym/logs/CAS02_ground/test/policies/policy_1.pt
joystick_use = True
joystick_opened = False



class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0

# if joystick_use:

#     pygame.init()

#     try:
#         # 获取手柄
#         joystick = pygame.joystick.Joystick(0)
#         joystick.init()
#         joystick_opened = True
#     except Exception as e:
#         print(f"无法打开手柄：{e}")

#     # 用于控制线程退出的标志
#     exit_flag = False


#     # 处理手柄输入的线程
#     def handle_joystick_input():
#         global exit_flag, cmd
        
        
#         while not exit_flag:
#             # 获取手柄输入
#             pygame.event.get()

#             # 更新机器人命令
#             cmd.vx = -joystick.get_axis(1) * 1.
#             cmd.vy = -joystick.get_axis(0) * 0.
#             cmd.dyaw = -joystick.get_axis(3) * 0.5

#             print(cmd.vx, cmd.vy , cmd.dyaw)

#             # 等待一小段时间，可以根据实际情况调整
#             pygame.time.delay(100)

#         # 启动线程

#     if joystick_opened and joystick_use:
#         joystick_thread = Thread(target=handle_joystick_input)
#         joystick_thread.start()

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    # return (q, dq, quat, v, omega, gvec)
    base_pos = q[:3]
    foot_positions = []
    foot_forces = []

    foot_forces.append(data.sensor('left_force').data.astype(np.double)[2]) 
    foot_forces.append(data.sensor('right_force').data.astype(np.double)[2])
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if '6_link' in body_name:  # 根据你的模型具体命名选择
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            # foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

startRecordFlag = False
start_ts = time.time()
dof_pos_buffer = []
dof_vel_buffer = []
root_states_buffer = []
euler_xyz_buffer = []
action_buffer = []
ts_buffer = []



def run_mujoco(policy, cfg, pos):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # if pose == "sit":
    #     init_pose = {
    #     # 左腿
    #         "leg_l1_joint": -1.3,  # 髋关节弯曲
    #         #"leg_l2_joint": 0.3,   # 侧摆
    #         #"leg_l3_joint": 0.1,   # 旋转
    #         "leg_l4_joint": 1.5,   # 膝关节
    #         #"leg_l5_joint": -0.5,  # 踝关节
    #         # 右腿（对称）
    #         "leg_r1_joint": -1.3,
    #         #"leg_r2_joint": -0.3,
    #         #"leg_r3_joint": -0.1,
    #         "leg_r4_joint": 1.5,
    #         #"leg_r5_joint": -0.5,
    #         # 手臂
    #         #"upper_left_1_joint": 0.5,
    #         #"upper_right_1_joint": 0.5
    #     }
    #     # 应用初始姿态
    #     for joint_name, angle in init_pose.items():
    #         joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    #         data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_step(model, data)

    # mujoco.set_mjcb_control(my_controller)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    #viewer.launch(model, data) 
    target_q = np.zeros((cfg.env.num_dofs), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    actions_scaled = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action_rescale = cfg.control.action_scale
    #print("比例1", action_rescale)
    

    rpy = np.zeros(3, dtype=np.double) # roll pitch yaw

    hist_obs = deque()
    for _ in range(cfg.env.num_actor_history):
        hist_obs.append(np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double))

    count_lowlevel = 0
    count_phase = 0

    default_dof_pos = np.array([-0.185, 0.0, 0.0, 0.36, -0.175,  0.0,  \
                                -0.185, 0.0, 0.0, 0.36, -0.175,  0.0,
                                0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0])


    render_index =0 

    global startRecordFlag
    global target_pos
    global action_buffer

    logger = MjLogger(cfg.sim_config.dt)
    stop_state_log = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    actuated_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,21]

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec , base_pos, foot_positions, foot_forces= get_obs(data,model)
        # q = q[-cfg.env.num_actions:] 
        # dq = dq[-cfg.env.num_actions:]
        # q 的长度是30，为啥我不知道
        q = np.array(data.actuator_length)
        dq = np.array(data.actuator_velocity)

        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces

        startRecordFlag = True
        # 200hz -> 50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            #missing update action scale
            last_action[:] = action[:]
            obs = np.zeros([1, cfg.env.num_one_step_observations], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi 
            obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, 3:6] = gvec
            obs[0, 6:25] = q[actuated_indices] * cfg.normalization.obs_scales.dof_pos
            obs[0, 25:44] = dq[actuated_indices] * cfg.normalization.obs_scales.dof_vel
            obs[0, 44:63] = action[:cfg.env.num_actions]
            obs[0, 63] = 0.25  # action_scale
            # obs *= count_lowlevel > 30
            # print(obs)

            rpy[0:2] = eu_ang[:2] 

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()
            # print("aaa",cfg.env.num_one_step_observations)
            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.num_actor_history):
                policy_input[0, i * cfg.env.num_one_step_observations : (i + 1) * cfg.env.num_one_step_observations] = hist_obs[i][0, :]
            action[:cfg.env.num_actions] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            if count_lowlevel < 750:
                target_q *= 0
            else:
                target_q[actuated_indices] = action * action_rescale 
            # print("比例2", action_rescale)
            # print("目标角度",target_pos)
            target_pos = target_q + q
            action_buffer.append(target_pos)

            
        action_rescale = np.clip((action_rescale - 0.02), 0.25, np.inf)
        # action *= count_lowlevel > 30
        target_dq = np.zeros((cfg.env.num_dofs), dtype=np.double)

        # if cfg.normalization.actions_filter:
        #     rate_ = (count_lowlevel % cfg.sim_config.decimation + 1.)/cfg.sim_config.decimation
        #     action_filter =  (1.- rate_) * last_action + rate_ * action
        #     target_q_filter = action_filter * cfg.control.action_scale
        # else:
        #     target_q_filter = action * cfg.control.action_scale

        tau = pd_control(target_q + q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit) # Clamp torques
        data.ctrl = tau
        applied_tau = tau #data.actuator_force
        idx = 5 
        dof_pos_target = target_q + default_dof_pos

        if _ < stop_state_log-1:
            logger.log_states(
            {   
                'base_height' : base_z,
                'foot_z_l' : foot_z[0],
                'foot_z_r' : foot_z[1],
                'contact_forces_z' : foot_force_z,
                'base_vel_x': v[0],
                'command_x': 0.5,
                'base_vel_y': v[1],
                'command_y': 0,
                'base_vel_z': v[2],
                'base_vel_yaw': omega[2],
                'command_yaw': 0,
                'dof_pos_target':  action[ :]*  cfg.control.action_scale ,
                'dof_pos': q[idx],
                'dof_vel': dq[idx],
                'dof_torque': applied_tau[idx],
                'cmd_dof_torque': tau[idx],
                'imu_p': rpy[0].item(),
                'imu_r': rpy[1].item(),
                'dof_pos_target[0]':  target_pos[0].item(),
                'dof_pos_target[1]':  target_pos[1].item(),
                'dof_pos_target[2]': target_pos[2].item(),
                'dof_pos_target[3]': target_pos[3].item(),
                'dof_pos_target[4]': target_pos[4].item(),
                'dof_pos_target[5]': target_pos[5].item(),
                'dof_pos_target[6]': target_pos[6].item(),
                'dof_pos_target[7]': target_pos[7].item(),
                'dof_pos_target[8]': target_pos[8].item(),
                'dof_pos_target[9]': target_pos[9].item(),
                'dof_pos_target[10]': target_pos[10].item(),
                'dof_pos_target[11]': target_pos[11].item(),
                'dof_pos_target[12]': target_pos[12].item(),
                'dof_pos_target[13]': target_pos[13].item(),
                'dof_pos_target[14]': target_pos[14].item(),
                'dof_pos_target[15]': target_pos[15].item(),
                'dof_pos_target[16]': target_pos[16].item(),
                'dof_pos_target[17]': target_pos[17].item(),
                'dof_pos_target[18]': target_pos[18].item(),
                'dof_pos_target[19]': target_pos[19].item(),
                'dof_pos_target[20]': target_pos[20].item(),
                'dof_pos_target[21]': target_pos[21].item(),
                'dof_pos_target[22]': target_pos[22].item(),
                'dof_pos':    q[0].item(),
                'dof_pos[0]': q[0].item(),
                'dof_pos[1]': q[1].item(),
                'dof_pos[2]': q[2].item(),
                'dof_pos[3]': q[3].item(),
                'dof_pos[4]': q[4].item(),
                'dof_pos[5]': q[5].item(),
                'dof_pos[6]': q[6].item(),
                'dof_pos[7]': q[7].item(),
                'dof_pos[8]': q[8].item(),
                'dof_pos[9]': q[9].item(),
                'dof_pos[10]': q[10].item(),
                'dof_pos[11]': q[11].item(),
                'dof_pos[12]': q[12].item(),
                'dof_pos[13]': q[13].item(),
                'dof_pos[14]': q[14].item(),
                'dof_pos[15]': q[15].item(),
                'dof_pos[16]': q[16].item(),
                'dof_pos[17]': q[17].item(),
                'dof_pos[18]': q[18].item(),
                'dof_pos[19]': q[19].item(),
                'dof_pos[20]': q[20].item(),
                'dof_pos[21]': q[21].item(),
                'dof_pos[22]': q[22].item(),
                'dof_torque': applied_tau[0].item(),
                'dof_torque[0]': applied_tau[0].item(),
                'dof_torque[1]': applied_tau[1].item(),
                'dof_torque[2]': applied_tau[2].item(),
                'dof_torque[3]': applied_tau[3].item(),
                'dof_torque[4]': applied_tau[4].item(),
                'dof_torque[5]': applied_tau[5].item(),
                'dof_torque[6]': applied_tau[6].item(),
                'dof_torque[7]': applied_tau[7].item(),
                'dof_torque[8]': applied_tau[8].item(),
                'dof_torque[9]': applied_tau[9].item(),
                'dof_torque[10]': applied_tau[10].item(),
                'dof_torque[11]': applied_tau[11].item(),
                'dof_torque[12]': applied_tau[12].item(),
                'dof_torque[13]': applied_tau[13].item(),
                'dof_torque[14]': applied_tau[14].item(),
                'dof_torque[15]': applied_tau[15].item(),
                'dof_torque[16]': applied_tau[16].item(),
                'dof_torque[17]': applied_tau[17].item(),
                'dof_torque[18]': applied_tau[18].item(),
                'dof_torque[19]': applied_tau[19].item(),
                'dof_torque[20]': applied_tau[20].item(),
                'dof_torque[21]': applied_tau[21].item(),
                'dof_torque[22]': applied_tau[22].item(),
                'dof_vel': dq[0].item(),
                'dof_vel[0]': dq[0].item(),
                'dof_vel[1]': dq[1].item(),
                'dof_vel[2]': dq[2].item(),
                'dof_vel[3]': dq[3].item(),
                'dof_vel[4]': dq[4].item(),
                'dof_vel[5]': dq[5].item(),
                'dof_vel[6]': dq[6].item(),
                'dof_vel[7]': dq[7].item(),
                'dof_vel[8]': dq[8].item(),
                'dof_vel[9]': dq[9].item(),
                'dof_vel[10]': dq[10].item(),
                'dof_vel[11]': dq[11].item(),
                'dof_vel[12]': dq[12].item(),
                'dof_vel[13]': dq[13].item(),
                'dof_vel[14]': dq[14].item(),
                'dof_vel[15]': dq[15].item(),
                'dof_vel[16]': dq[16].item(),
                'dof_vel[17]': dq[17].item(),
                'dof_vel[18]': dq[18].item(),
                'dof_vel[19]': dq[19].item(),
                'dof_vel[20]': dq[20].item(),
                'dof_vel[21]': dq[21].item(),
                'dof_vel[22]': dq[22].item(),
            }
            )
        else:
            logger._plot(save_path="/home/casbot/ZHZ_ws/HoST/legged_gym/legged_gym/plots/fig4.png")

        time.sleep(0.03)
        viewer.render()

        mujoco.mj_step(model, data)

        render_index +=1
        count_lowlevel += 1

        if cmd.vx > 0.1:
            count_phase += 1
        else:
            cmd.vx = 0.

    startRecordFlag = False
    now = time.time()
    ts_diff = now-start_ts  
    print("total time (ms) ", ts_diff*1000)

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    if args.terrain:
        pose = "sit"
    else:
        pose = "sit"

    class Sim2simCfg(CAS02CfgGround19):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/02/mjmodel02fb.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/02/02_sit_chair.xml'
            sim_duration = 10.0
            # low level control frequency (In sim 200Hz, in real robot 500Hz)
            dt = 0.002
            # policy inference frequency 50Hz
            decimation = 10

        class robot_config:
            kps = np.array([350, 350, 350, 350, 100, 100, \
                            350, 350, 350, 350, 100, 100,
                            200,
                            200, 200, 200, 200, 200,
                            200, 200, 200, 200, 200], dtype=np.double)
            kds = np.array([4.0, 4.0, 4.0, 4.0, 2.0, 2.0,  \
                            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
                            4.0,
                            4.0, 4.0, 2.0, 4.0, 2.0,
                            4.0, 4.0, 2.0, 4.0, 2.0], dtype=np.double)
        
            # tau_limit = np.array([120., 120., 120., 120.,  90.,  64.,   \
            #                       120., 120., 120., 120.,  90.,  64.], dtype=np.double)
            tau_limit = np.array([150., 150., 150., 250.,  70.,  70.,   \
                                  150., 150., 150., 250.,  70.,  70.,
                                  70,
                                  65., 65., 16., 65., 16.,
                                  65., 65., 16., 65., 16.], dtype=np.double)
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), pose)
