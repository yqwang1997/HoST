# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class MjLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        # self.plot_process = Process(target=self._plot)
        # self.plot_process.start()
        self._plot()

    def _plot(self):
        nb_rows = 6
        nb_cols = 7
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log

        # plot joint targets and measured positions

        if log["dof_pos"]: 
            log["dof_pos"] = np.array(log["dof_pos"])
            log["dof_pos_target"] = np.array(log["dof_pos_target"])
            for i in range(12):
                if i < 6:
                    a = axs[i, 0]
                else:
                    a = axs[i-6, 1]
                        
                a.plot(time, log["dof_pos["+str(i)+"]"], label='measured')
    
            
                a.plot(time, log["dof_pos_target["+str(i)+"]"], label='target')
                a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position '+ str(i))
                a.legend()

        # plot joint velocity
        if log["dof_vel"]: 
            log["dof_vel"] = np.array(log["dof_vel"])
            for i in range(12):
                if i < 6:
                    a = axs[i, 2]
                else:
                    a = axs[i-6, 3]
                a.plot(time, log["dof_vel["+str(i)+"]"], label='dof_vel')
                a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity '+ str(i))
                a.legend()

        # plot torques
        if log["dof_torque"]: 
            log["dof_torque"] = np.array(log["dof_torque"])
            for i in range(12):
                if i < 6:
                    a = axs[i, 4]
                else:
                    a = axs[i-6, 5]
                a.plot(time, log["dof_torque["+str(i)+"]"], label='dof_torque')
                a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque '+ str(i))
                a.legend()   

        # plot base vel x
        a = axs[0, 6]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()

        # plot base vel y
        a = axs[1, 6]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()

        # plot base vel yaw
        a = axs[2, 6]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()

        # plot base vel z
        a = axs[3, 6]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()

        # plot contact forces
        a = axs[4, 6]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()

        # plot imu
        # a = axs[4, 6]
        # if log["imu_p"]:
        #     pitch = np.array(log["imu_p"])
        #     a.plot(time, pitch, label=f'pitch ')
        # a.set(xlabel='time [s]', ylabel='Imu P [Rad]', title='IMU pitch')
        # a.legend()
        a = axs[5, 6]
        if log["imu_r"]:
            roll = np.array(log["imu_r"])
            a.plot(time, roll, label=f'roll ')
        a.set(xlabel='time [s]', ylabel='Imu R [Rad]', title='IMU Roll')
        a.legend()
        
            # 创建独立窗口显示所有23个关节扭矩（分成小图）
        if any(f"dof_torque[{i}]" in self.state_log for i in range(23)):
            # 使用4行6列布局，与主窗口风格一致
            nb_rows_torque = 4
            nb_cols_torque = 6
            
            fig_torque, axs_torque = plt.subplots(nb_rows_torque, nb_cols_torque, figsize=(16, 10))
            fig_torque.suptitle('All 23 Joint Torques', fontsize=16)
            
            for i in range(23):
                  row = i // nb_cols_torque
                  col = i % nb_cols_torque
                  
                  key = f"dof_torque[{i}]"
                  if key in self.state_log:
                        axs_torque[row, col].plot(time, self.state_log[key], 
                                                label='dof_torque',
                                                color='#1f77b4',
                                                linewidth=1.0)
                        axs_torque[row, col].set(xlabel='time [s]', 
                                          ylabel='Joint Torque [Nm]', 
                                          title=f'Torque {i}')
                        axs_torque[row, col].legend(fontsize=8)
                        axs_torque[row, col].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(23, nb_rows_torque * nb_cols_torque):
                  row = i // nb_cols_torque
                  col = i % nb_cols_torque
                  axs_torque[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.show(block=False)


        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()