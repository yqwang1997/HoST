import sys
import os
import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
import math

# 直接导入torch（避免Isaac Gym导入问题）
import torch

# 导入MuJoCo
import mujoco
try:
    import mujoco_python_viewer as mujoco_viewer
    print("使用 mujoco_python_viewer")
except ImportError:
    try:
        import mujoco_viewer
        print("使用 mujoco_viewer")
    except ImportError:
        print("警告：没有找到viewer包，将无法可视化")
        mujoco_viewer = None

# 手动实现get_args功能，避免导入Isaac Gym
import argparse

def get_args():
    """简化的参数解析，避免Isaac Gym依赖"""
    parser = argparse.ArgumentParser(description='MuJoCo sim2sim testing')
    parser.add_argument('--model_path', type=str, default='/home/xi-lap/Documents/HoST_CASBOT/legged_gym/logs/CAS02_ground/exported/policies/policy_cas02_12_4000.pt', help='Path to trained model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--xml_path', type=str, default='/home/xi-lap/Documents/HoST_CASBOT/legged_gym/resources/robots/02/CASBOT_02_up.xml', help='Path to MuJoCo XML file')
    parser.add_argument('--task', type=str, default='cas02', help='Task name')
    
    return parser.parse_args()

# 简化的ActorCritic网络定义
class SimpleActorCritic(torch.nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.actor(x)
    
    def act_inference(self, x):
        return self.forward(x)

class MuJoCoTester:
    def __init__(self, model_path, xml_path, device='cpu'):
        """
        初始化MuJoCo测试器
        """
        self.device = device
        
        # 加载MuJoCo模型
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        # 设置时间步长
        self.mj_model.opt.timestep = 0.005
        
        # 从模型文件推断参数
        self.num_obs = 76 * 6  # 假设的观测维度
        self.num_actions = 23   # 假设的动作维度
        self.decimation = 4
        
        # 初始化观测历史
        self.obs_history = deque()
        for _ in range(6):
            self.obs_history.append(np.zeros([1, 76], dtype=np.double))
        
        # 加载策略网络
        self.load_policy(model_path)
        
        # 初始化
        self.reset_simulation()
        self.count_lowlevel = 1
        self.last_actions = np.zeros(self.num_actions)
        
        # 观测缩放参数
        self.obs_scales = {
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
            'lin_vel': 2.0
        }
        
    def load_policy(self, model_path):
        """加载策略网络"""
        try:
            print(f"尝试加载模型: {model_path}")
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 根据警告信息，直接使用torch.jit.load加载JIT模型
            if model_path.endswith('.pt'):
                print("检测到JIT模型格式，使用torch.jit.load加载")
                self.policy_net = torch.jit.load(model_path, map_location=self.device)
                print("JIT模型加载成功")
                
                # 检查JIT模型的方法
                available_methods = [method for method in dir(self.policy_net) if not method.startswith('_')]
                print(f"JIT模型可用方法: {available_methods}")
                
            else:
                # 尝试加载checkpoint
                print("尝试加载checkpoint格式")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print("检测到训练checkpoint格式")
                    # 需要从checkpoint中获取网络参数信息
                    state_dict = checkpoint['model_state_dict']
                    
                    # 尝试从state_dict推断网络结构
                    actor_layers = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k]
                    if actor_layers:
                        # 获取输入输出维度
                        first_layer = f'actor.0.weight'
                        last_layer = max([k for k in actor_layers if '.weight' in k])
                        
                        if first_layer in state_dict:
                            self.num_obs = state_dict[first_layer].shape[1]
                        if last_layer in state_dict:
                            self.num_actions = state_dict[last_layer].shape[0]
                        
                        print(f"从checkpoint推断: obs_dim={self.num_obs}, action_dim={self.num_actions}")
                
                    # 创建网络并加载权重
                    self.policy_net = SimpleActorCritic(self.num_obs, self.num_actions).to(self.device)
                    self.policy_net.load_state_dict(state_dict)
                    print("Checkpoint加载成功")
                else:
                    raise ValueError("未知的模型格式")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"错误详细信息: {type(e).__name__}")
            
            # 尝试检查模型文件信息
            try:
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                    print(f"模型文件大小: {file_size:.2f} MB")
                    
                    # 尝试直接用torch.jit.load
                    print("强制尝试JIT加载...")
                    self.policy_net = torch.jit.load(model_path, map_location=self.device)
                    print("强制JIT加载成功！")
                    return
            except Exception as e2:
                print(f"强制JIT加载也失败: {e2}")
            
            print("创建随机策略网络用于测试")
            self.policy_net = SimpleActorCritic(self.num_obs, self.num_actions).to(self.device)
        
        self.policy_net.eval()
        print("策略网络加载完成")
      
    def reset_simulation(self):
        """重置仿真"""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_initial_pose()
        
    def _set_initial_pose(self):
        """设置初始姿态"""
        # 设置基座位置
        self.mj_data.qpos[0:3] = [0.0, 0.0, 0.3]
        self.mj_data.qpos[3:7] = [1.0, 0.0, -1.0, 0.0]  # 四元数 w,x,y,z
        
        # 设置关节角度
        if len(self.mj_data.qpos) > 7:
            joint_angles = np.array([
                -0.185, 0, 0, 0.36, -0.175, 0,  # 左腿
                -0.185, 0, 0, 0.36, -0.175, 0,  # 右腿
                0.,  # 腰部
                0., 0., 0., 0., 0.,  # 左臂
                0., 0., 0., 0., 0.   # 右臂
            ])
            
            # 只设置存在的关节
            n_joints = min(len(joint_angles), len(self.mj_data.qpos) - 7)
            self.mj_data.qpos[7:7+n_joints] = joint_angles[:n_joints]
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def get_obs(self):
        """获取基本观测"""
        q = self.mj_data.qpos.copy()
        dq = self.mj_data.qvel.copy()
        
        # 基座姿态和角速度
        quat = q[3:7]  # [w, x, y, z]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])  # 转为[x,y,z,w]
        omega = dq[3:6]
        
        # 关节部分
        q_joints = q[7:] if len(q) > 7 else np.array([])
        dq_joints = dq[6:] if len(dq) > 6 else np.array([])
        
        return q_joints, dq_joints, quat_scipy, omega
    
    def get_isaac_gym_obs(self):
        """构造观测"""
        q, dq, quat, omega = self.get_obs()
        
        obs = np.zeros([1, 76], dtype=np.float32)
        
        # 角速度
        if len(omega) >= 3:
            r = R.from_quat(quat)
            omega_body = r.apply(omega, inverse=True)
            obs[0, 0:3] = omega_body * self.obs_scales['ang_vel']
            
            # 重力向量
            gvec = r.apply(np.array([0., 0., -1.]), inverse=True)
            obs[0, 3:6] = gvec
        
        # 关节位置和速度
        if len(q) > 0:
            default_pos = np.zeros_like(q)
            obs[0, 6:6+len(q)] = (q - default_pos) * self.obs_scales['dof_pos']
        
        if len(dq) > 0:
            obs[0, 29:29+len(dq)] = dq * self.obs_scales['dof_vel']
        
        # 上一步动作
        obs[0, 52:52+len(self.last_actions)] = self.last_actions
        
        return obs
    
    def step_simulation(self, actions):
        """执行仿真步骤"""
        self.last_actions = actions.copy()
        
        # 简单位置控制
        if hasattr(self.mj_data, 'ctrl') and len(self.mj_data.ctrl) > 0:
            n_ctrl = min(len(actions), len(self.mj_data.ctrl))
            self.mj_data.ctrl[:n_ctrl] = actions[:n_ctrl]
        
        for _ in range(self.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)
    
    def run_test(self, duration=10.0, render=True):
        """运行测试"""
        dt = self.mj_model.opt.timestep * self.decimation
        steps = int(duration / dt)
        
        viewer = None
        if render and mujoco_viewer is not None:
            try:
                viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
            except Exception as e:
                print(f"无法创建viewer: {e}")
                viewer = None
        
        print(f"开始仿真，总步数: {steps}")
        
        # 添加调试标志，只打印一次错误信息
        debug_printed = False
        
        for step in range(steps):
            if self.count_lowlevel % self.decimation == 0:
                # 获取观测
                current_obs = self.get_isaac_gym_obs()
                self.obs_history.append(current_obs)
                self.obs_history.popleft()
                
                # 构造策略输入
                policy_input = np.zeros([1, self.num_obs], dtype=np.float32)
                for i in range(6):
                    policy_input[0, i * 76:(i + 1) * 76] = self.obs_history[i][0, :]
                
                # 获取动作
                with torch.no_grad():
                    try:
                        input_tensor = torch.tensor(policy_input).to(self.device)
                        
                        # 尝试不同的调用方式
                        if hasattr(self.policy_net, 'act_inference'):
                            actions = self.policy_net.act_inference(input_tensor)
                        elif hasattr(self.policy_net, 'forward'):
                            actions = self.policy_net.forward(input_tensor)
                        else:
                            # 直接调用JIT模型
                            actions = self.policy_net(input_tensor)
                        
                        actions = actions.squeeze(0).cpu().numpy()
                        actions = np.clip(actions, -1, 1)
                        
                    except Exception as e:
                        if not debug_printed:
                            print(f"策略推理错误: {e}")
                            print(f"输入张量形状: {input_tensor.shape}")
                            print(f"模型类型: {type(self.policy_net)}")
                            available_methods = [method for method in dir(self.policy_net) if not method.startswith('_')]
                            print(f"可用方法: {available_methods}")
                            debug_printed = True
                        
                        actions = np.zeros(self.num_actions)
                
                # 执行动作
                self.step_simulation(actions)
            else:
                mujoco.mj_step(self.mj_model, self.mj_data)
            
            if viewer:
                try:
                    viewer.render()
                except:
                    pass
            
            self.count_lowlevel += 1
            
            # 进度输出
            if step % 100 == 0:
                base_pos = self.mj_data.qpos[:3]
                print(f"Step {step}/{steps}: Base height: {base_pos[2]:.3f}m")
        
        if viewer:
            try:
                viewer.close()
            except:
                pass

def main():
    """主函数"""
    args = get_args()
    
    model_path = args.checkpoint_path or args.model_path
    xml_path = args.xml_path
    
    if not model_path:
        raise ValueError("Must specify --model_path or --checkpoint_path")
    if not xml_path:
        raise ValueError("Must specify --xml_path")
    
    print(f"加载模型: {model_path}")
    print(f"加载MuJoCo模型: {xml_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        tester = MuJoCoTester(model_path, xml_path, device=device)
        print("开始测试...")
        tester.run_test(duration=20.0, render=True)
        print("测试完成")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()