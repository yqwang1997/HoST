# HoST 工作文档
## 参数
*/HoST/legged_gym/legged_gym/envs/CAS02/CAS02_config_ground.py*
### kp kd 
```
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'leg_pelvic_pitch': 300, # 350
                     'leg_pelvic_roll':300, # 350
                     'leg_pelvic_yaw':300, # 350
                     'knee': 300, # 350 
                     'ankle': 100, # 120
                     'shoulder_pitch': 160, # 200 
                     'shoulder_roll': 160, # 200
                     'shoulder_yaw': 60, # 200
                     'elbow': 120, # 350
                     'waist': 120, # 200
                     'wrist_yaw': 60, # 100
                     }  # [N*m/rad]
        damping = {  'leg_pelvic_pitch': 7, # 4
                     'leg_pelvic_roll': 7,
                     'leg_pelvic_yaw': 7,
                     'knee': 6, # 4
                     'ankle': 5, # 2
                     'shoulder_pitch': 4,
                     'shoulder_roll': 4,
                     'shoulder_yaw': 4,
                     'elbow': 4, # 5
                     'waist': 4,
                     'wrist_yaw': 4, # 5
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
```
修改髋关节包含的pitch、roll、yaw，单独命名和设置pd参数。  
对于所有的pd参数，因为simtosim的抖动明显所以把kp改小，kd改大。
### 目标高度设置
```
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.92
        target_head_height = 1.3
        target_head_margin = 1.1
        target_base_height_phase1 = 0.55
        target_base_height_phase2 = 0.55
        target_base_height_phase3 = 0.8
        orientation_threshold = 0.9


    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 1.3
        target_head_margin = 1.1
        orientation_height_threshold = 0.9
        target_base_height = 0.92
```
针对cas02 在 isaacgym 中的实际高度，对头部和基座的目标高度进行修改。
### post_task
```
post_task = False
```
这个参数设置为false比较好，因为如果设置为ture，在站立阶段，精细的姿态相关reward值会被覆盖掉，都设置为1。  
这会影响 _reward_shank_orientation 和 _reward_ground_parallel 在站立阶段的姿态引导。

## REWARD
*/HoST/legged_gym/legged_gym/envs/base/host_ground.py*  
### 髋关节
```
    def _reward_hip_yaw_deviation(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 0.9 ) | (torch.min(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 0.5)
        return reward

    def _reward_hip_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 0.9 ) | (torch.min(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 0.5)
        return reward
```
由于策略的起身姿态偏向于使用髋关节从roll方向直膝横向夹起，减小髋关节设置的最大值，避免这种不实际的起身姿态。
### 脚部位置
```
    style_left_foot_displacement = 3 # 2.5
    style_right_foot_displacement = 3 # 2.5

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
```
鼓励脚部位置与root位置有一定距离，调大reward权重，避免双脚距离太近，让机器人稳定站立。
### 脚部与地面接触
```
    def _reward_ground_parallel(self):
        left_ankle_pos = self.rigid_body_states[:, self.left_ankle_indices, 2].clone() * 10
        right_ankle_pos = self.rigid_body_states[:, self.right_ankle_indices, 2].clone() * 10
        var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
        var = torch.mean(torch.concat([left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1), dim=-1)
        reward = var < 0.02 # 0.05
```
针对机器人站立之后脚部接触地面不充分的问题，减小两只脚之间的z方向方差限制，改善踮脚问题。
### 起身姿态
```
    def _reward_knee_bend_enforcement(self):
        """
        在起身过程中强制膝关节保持弯曲，避免直膝夹起
        只在起身阶段生效，站立后取消此限制
        """
        # 获取左右膝关节角度
        left_knee_pos = self.dof_pos[:, self.left_knee_joint_indices]
        right_knee_pos = self.dof_pos[:, self.right_knee_joint_indices]
        
        # 计算两个膝关节的最小弯曲角度（取较小值，确保两腿都弯曲）
        min_knee_bend = torch.min(torch.cat([left_knee_pos, right_knee_pos], dim=1), dim=1)[0]
        
        # 判断是否在起身阶段（基于基座高度）
        base_height = self.root_states[:, 2]
        in_getup_phase = base_height < self.cfg.rewards.target_base_height_knee_limit
        
        # 判断是否处于头部抬起但身体未完全站立的关键起身阶段
        head_height = self.rigid_body_states[:, self.head_indices, 2].squeeze(1)
        head_lifted = head_height > self.cfg.rewards.head_lift_threshold  # 使用配置参数
        critical_getup_phase = in_getup_phase & head_lifted
        
        # 计算膝关节弯曲奖励
        knee_bend_reward = tolerance(
            min_knee_bend, 
            [self.cfg.rewards.knee_bend_threshold, np.inf], 
            self.cfg.rewards.knee_bend_margin, 
            0.1
        )
        
        # 对于完全直膝的情况给予额外惩罚
        straight_knee_penalty = (min_knee_bend < 0.1).float()
        
        # 综合奖励：鼓励弯曲 - 惩罚直膝
        total_reward = knee_bend_reward - straight_knee_penalty * 2.0
        
        # 只在关键起身阶段应用此奖励
        return total_reward * critical_getup_phase.float()
```
针对起身过程中容易直膝夹起、影响simtoreal部署性能的问题，加入reward鼓励起身过程中膝盖弯曲，惩罚直膝起身。  
加入头部高度判断，保证此项reward仅在起身过程中段发挥作用，不影响最初平躺和最终站立的动作探索效果。
### 站立姿态（双腿前后岔开）
```
    def _reward_feet_parallel_alignment(self):
        """专门奖励左右脚平行对齐，避免前后岔开"""
        base_pos = self.root_states[:, :3].clone()
        base_quat = self.root_states[:, 3:7].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].squeeze(1)
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].squeeze(1)
        
        left_foot_world = left_foot_pos - base_pos
        right_foot_world = right_foot_pos - base_pos
        left_foot_body = quat_rotate_inverse(base_quat, left_foot_world)
        right_foot_body = quat_rotate_inverse(base_quat, right_foot_world)
        
        forward_diff = torch.abs(left_foot_body[:, 0] - right_foot_body[:, 0])
        
        # 构造奖励：前后差异越小奖励越高
        alignment_error = torch.square(forward_diff).clamp(0.08, np.inf)
        reward = torch.exp(alignment_error * self.cfg.rewards.left_foot_displacement_sigma)
        
        left_on_ground = (self.rigid_body_states[:, self.left_foot_indices, 2] < 0.3).squeeze(1)
        right_on_ground = (self.rigid_body_states[:, self.right_foot_indices, 2] < 0.3).squeeze(1)
        both_feet_on_ground = left_on_ground & right_on_ground
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        
        # 添加头部高度条件判断
        head_height = self.rigid_body_states[:, self.head_indices, 2].squeeze(1)
        head_high_enough = head_height > 1.28
        
        return reward * both_feet_on_ground * standup * head_high_enough
```
针对机器人站立之后双腿前后岔开影响站立稳定性的问题，计算双脚基于机体坐标系的前后距离，鼓励双脚横向平行。  
加入头部高度条件判断，减少此项reward对起身过程的影响，仅在起身后期和站立时期改善姿态。
### 站立姿态（站立稍微倾斜，重心向一侧靠）
**新加入的reward，待完善**  
```
    def _reward_center_of_mass_stability(self):
        """
        基于脚部压力分布判断重心稳定性
        """
        # 获取左右脚的接触力
        left_foot_contact = self.contact_forces[:, self.left_foot_indices, 2]  # Z方向接触力
        right_foot_contact = self.contact_forces[:, self.right_foot_indices, 2]
        
        left_foot_force = torch.sum(torch.abs(left_foot_contact), dim=1)
        right_foot_force = torch.sum(torch.abs(right_foot_contact), dim=1)
        
        # 计算重心分布比例
        total_force = left_foot_force + right_foot_force + 1e-8  # 避免除零
        left_ratio = left_foot_force / total_force
        right_ratio = right_foot_force / total_force
        
        # 理想情况下左右脚应该各承担50%重量
        ideal_ratio = 0.5
        ratio_error = torch.abs(left_ratio - ideal_ratio) + torch.abs(right_ratio - ideal_ratio)
        
        # 只在站立阶段应用
        base_height = self.root_states[:, 2]
        standing_phase = base_height > self.cfg.rewards.target_base_height_phase3
        
        # 重心越平衡奖励越高
        stability_reward = torch.exp(-ratio_error * self.cfg.rewards.com_stability_weight)
        
        return stability_reward * standing_phase.float()
    
    def _reward_left_right_symmetry(self):
        """
        鼓励左右腿关节角度的对称性，避免重心偏移
        """
        # 根据URDF关节顺序正确获取关节角度
        left_hip_pitch = self.dof_pos[:, 0:1]      # left_leg_pelvic_pitch_joint
        right_hip_pitch = self.dof_pos[:, 6:7]     # right_leg_pelvic_pitch_joint
        
        left_hip_roll = self.dof_pos[:, 1:2]       # left_leg_pelvic_roll_joint  
        right_hip_roll = self.dof_pos[:, 7:8]      # right_leg_pelvic_roll_joint
        
        left_knee = self.dof_pos[:, 3:4]           # left_leg_knee_pitch_joint
        right_knee = self.dof_pos[:, 9:10]         # right_leg_knee_pitch_joint
        
        left_ankle_pitch = self.dof_pos[:, 4:5]    # left_leg_ankle_pitch_joint
        right_ankle_pitch = self.dof_pos[:, 10:11] # right_leg_ankle_pitch_joint
        
        left_ankle_roll = self.dof_pos[:, 5:6]     # left_leg_ankle_roll_joint
        right_ankle_roll = self.dof_pos[:, 11:12]  # right_leg_ankle_roll_joint
        
        # 计算左右对称性误差
        hip_pitch_diff = torch.abs(left_hip_pitch - right_hip_pitch)
        hip_roll_diff = torch.abs(left_hip_roll + right_hip_roll)  # roll应该相反对称
        knee_diff = torch.abs(left_knee - right_knee)
        ankle_pitch_diff = torch.abs(left_ankle_pitch - right_ankle_pitch)
        ankle_roll_diff = torch.abs(left_ankle_roll + right_ankle_roll)  # roll也应该相反对称
        
        # 综合对称性误差
        total_asymmetry = (hip_pitch_diff + hip_roll_diff + knee_diff + 
                        ankle_pitch_diff + ankle_roll_diff).squeeze(1)
        
        # 只在站立阶段应用
        base_height = self.root_states[:, 2]
        standing_phase = base_height > self.cfg.rewards.target_base_height_phase3  # 0.8m
        
        # 使用tolerance函数：越对称奖励越高
        symmetry_reward = tolerance(total_asymmetry, [0, 0.2], 0.15, 0.1)
        
        return symmetry_reward * standing_phase.float()
```
针对站立之后腿部和脚部的抖动问题，观察测试表现和数据怀疑是因为站立之后的中心不在理想位置，需要很多实时调整。  
加入脚部压力reward和左右对称性reward，希望引导机器人站立的时候两只脚的受力平均一些，引导左右对称也是希望站立稳定。  
这两项reward是新加入的，还没有调试好，待改善。

## 效果较好的策略
Env:*/HoST/legged_gym/legged_gym/envs/base/host_ground_cas30.py*  
Config:*/HoST/legged_gym/tested_models_log/Jul17_17-47-01_train_cas02_30/CAS02_config_ground.py*  
isaacgym model:*/HoST/legged_gym/tested_models_log/Jul17_17-47-01_train_cas02_30/model_8000.pt*  
mujoco model:*/HoST/legged_gym/tested_models_log/exported/policies/policy_cas02_30_8000.pt*  