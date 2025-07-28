import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('/home/casbot/ZHZ_ws/HoST/legged_gym/resources/robots/02/02_sit_chair.xml')
data = mujoco.MjData(model)

init_pose = {
    # 左腿
    "leg_l1_joint": -1.3,  # 髋关节弯曲
    #"leg_l2_joint": 0.3,   # 侧摆
    #"leg_l3_joint": 0.1,   # 旋转
    "leg_l4_joint": 1.5,   # 膝关节
    #"leg_l5_joint": -0.5,  # 踝关节
    # 右腿（对称）
    "leg_r1_joint": -1.3,
    #"leg_r2_joint": -0.3,
    #"leg_r3_joint": -0.1,
    "leg_r4_joint": 1.5,
    #"leg_r5_joint": -0.5,
    # 手臂
    #"upper_left_1_joint": 0.5,
    #"upper_right_1_joint": 0.5
}

# 应用初始姿态
for joint_name, angle in init_pose.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = angle


# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()