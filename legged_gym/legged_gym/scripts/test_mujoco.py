import mujoco
from mujoco import viewer  # 直接从 mujoco 导入

xml_model = """
<mujoco>
    <worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="box_and_sphere" pos="0 0 0.2">
            <joint type="free"/>
            <geom name="red_box" type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
            <geom name="green_sphere" type="sphere" size=".1" pos=".2 0 0" rgba="0 1 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_model)
data = mujoco.MjData(model)

# 使用 mujoco 内置的 Viewer
viewer = viewer.launch(model, data)  # 注意方法名是 launch

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()  # 同步渲染数据

# 关闭查看器（通常按 ESC 键退出）
