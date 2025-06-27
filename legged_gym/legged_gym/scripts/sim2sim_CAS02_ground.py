import math
import numpy as np
import mujoco
from mujoco import viewer  # 直接从 mujoco 导入
print("viewer 的类型:", type(viewer))
print("viewer 的方法:", dir(viewer))
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym.envs.CAS02.CAS02_config_ground import CAS02Cfg as CAS02CfgGround
#from legged_gym.utils import  Logger
import torch

# 运行脚本
# python your_script.py --load_model /data/policy.pt

# x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# # x_scale, y_scale, yaw_scale = 2.5, 2.0, 0.0
    
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
    base_pos = q[:3]
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if '6_link' in body_name:  # 根据你的模型具体命名选择
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp - dq * kd
    return torque_out


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    # 从XML配置文件中创建MuJoCo模型对象。这通常包含了仿真的所有物理特性和对象定义。
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    # 设置模型的时间步长，这个时间步长决定了物理仿真的精度和速度。
    model.opt.timestep = cfg.sim_config.dt
    
    # 使用模型创建仿真数据对象，这个对象用于存储每一步仿真的状态数据。
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model
    print(len(cfg.robot_config.default_dof_pos))
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    # data.qpos[2] = 0.1
    # 执行一步物理仿真。这会根据当前的状态和模型定义来更新仿真数据（例如，物体的位置和速度）。
    mujoco.mj_step(model, data)
    
    # 创建一个视图器对象，用于可视化仿真。这使得用户可以看到仿真环境和其中的物体如何随时间演变。
    # viewer = mujoco_viewer.MujocoViewer(model, data) 老版本引用方法
    viewer.launch(model, data) 

    # 初始化目标关节角度数组，这里全部设置为零。这个数组用于定义期望的控制目标状态。
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
   
    # 初始化动作数组，同样全部设置为零。在强化学习或控制任务中，这个数组将被用来存储每一步的控制命令。
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    # 初始化一个历史观测的双端队列，用于存储一定数量的过去观测数据。这对于需要考虑时间序列信息的任务特别有用。
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    # 初始化一个计数器，用于跟踪低级控制循环的迭代次数。
    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    
    stop_state_log = 12000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        
        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
            obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang
            # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)


            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            # arr = policy_input[0]
            # arr_reshaped = arr.reshape((15, 47))
            
            # # 遍历这个二维数组，打印每一行
            # for row in arr_reshaped:
            #     print(row)
            
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            # print("\n", action)
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
        data.ctrl = tau
        applied_tau = data.actuator_force
        
        print("\ntau", tau)
        print("q", q)

        mujoco.mj_step(model, data)
        # viewer.render()
        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + cfg.robot_config.default_dof_pos
        if _ < stop_state_log:
            logger.log_states(
                {   
                    'base_height' : base_z,
                    'foot_z_l' : foot_z[0],
                    'foot_z_r' : foot_z[1],
                    'foot_forcez_l' : foot_force_z[0],
                    'foot_forcez_r' : foot_force_z[1],
                    'base_vel_x': v[0],
                    'command_x': x_vel_cmd,
                    'base_vel_y': v[1],
                    'command_y': y_vel_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': dof_pos_target[idx] ,
                    'dof_pos': q[idx],
                    'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx],
                    'cmd_dof_torque': tau[idx],
                    'dof_pos_target[0]': dof_pos_target[0].item(),
                    'dof_pos_target[1]': dof_pos_target[1].item(),
                    'dof_pos_target[2]': dof_pos_target[2].item(),
                    'dof_pos_target[3]': dof_pos_target[3].item(),
                    'dof_pos_target[4]': dof_pos_target[4].item(),
                    'dof_pos_target[5]': dof_pos_target[5].item(),
                    'dof_pos_target[6]': dof_pos_target[6].item(),
                    'dof_pos_target[7]': dof_pos_target[7].item(),
                    'dof_pos_target[8]': dof_pos_target[8].item(),
                    'dof_pos_target[9]': dof_pos_target[9].item(),
                    'dof_pos_target[10]': dof_pos_target[10].item(),
                    'dof_pos_target[11]': dof_pos_target[11].item(),
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
                }
                )
        
        elif _== stop_state_log:
            logger.print_rewards()

    viewer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(CAS02CfgGround):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'/home/casbot/ZHZ_ws/HoST/legged_gym/resources/robots/02/CASBOT_02.xml'
            else:
                mujoco_model_path = f'/home/casbot/ZHZ_ws/HoST/legged_gym/resources/robots/02/CASBOT_02.xml'
            sim_duration = 120.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([100, 100, 50, 100, 40, 40, 100, 100, 50, 100, 40, 40], dtype=np.double)
            kds = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=np.double)
            tau_limit = 350. * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([-0.185, 0., 0., 0.36, -0.175, 0., -0.185, 0., 0., 0.36, -0.175, 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., -0., 0.])


    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())