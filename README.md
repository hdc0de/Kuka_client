# Real Robot Policy Evaluation: Client-Server Architecture

## 背景

在真机 (KUKA iiwa + Robotiq 85) 上验证 sim-to-real 训练的 policy（ACT / Pi0.5 / Diffusion Policy 等）。

---

## 系统架构

```
主电脑 (Ubuntu 24.04, GPU)              ROS 机器 (连 KUKA + RealSense)
┌──────────────────────────┐            ┌──────────────────────────────┐
│  server.py               │   ZMQ      │  ros_bridge.py               │
│  Policy 推理 (GPU)        │◄──────────►│  - RealSense 相机采集          │
│  ZMQ REP :5555           │  REQ-REP   │  - ROS topic 读关节/发指令      │
│                          │            │  - 本地 action buffer         │
│  支持 policy:             │            │  - ZMQ REQ → server (按需)    │
│    random / go_home      │            └──────────────────────────────┘
│    act / pi05 / dp       │
└──────────────────────────┘
```

`ros_bridge.py` 是**自包含单文件**（内嵌 msg_utils + camera 逻辑），直接 scp 到 ROS 机器即可运行，无需安装本项目。

`client.py` 保留用于主电脑本地测试（DummyRobot + DummyCamera 或主电脑直连相机）。

---

## 目录结构

```
real_robot/
├── PLAN.md                          # 本文件
├── config.yaml                      # 主配置（greycup 任务）
├── config_ketchup_lite.yaml         # 番茄酱任务配置
├── server.py                        # Policy 推理服务（主电脑）
├── ros_bridge.py                    # ROS bridge，自包含单文件（ROS 机器）
├── client.py                        # 本地测试 client（主电脑，可选）
├── robot_interface.py               # 机器人抽象层（client.py 使用）
├── camera.py                        # 相机采集（client.py 使用）
├── msg_utils.py                     # ZMQ msgpack 序列化工具
├── test_stage2.py                   # 无硬件集成测试（in-process server+client）
├── plot_states_compare.py           # 真机 vs 训练数据关节轨迹对比
├── policies/
│   ├── __init__.py
│   ├── base_policy.py               # Policy 抽象基类 + RandomPolicy + GoHomePolicy
│   ├── act_policy.py                # LeRobot ACT 推理封装
│   ├── pi05_policy.py               # OpenPI Pi0.5 (JAX) 推理封装
│   └── dp_policy.py                 # Diffusion Policy (PyTorch) 推理封装
└── debug_output/                    # episode 录制数据（--save 时生成）
    └── ep{N}/
        ├── states.npy               # (T, 8) 关节+夹爪状态序列
        ├── actions.npy              # (T, 8) 发出的动作序列
        ├── base_camera.mp4          # base 相机视频
        ├── wrist_camera.mp4         # 手腕相机视频
        └── query_frames.mp4         # 推理帧拼图（base + wrist 并排）
```

---

## 实现状态

| # | 文件/功能 | 状态 | 说明 |
|---|----------|------|------|
| 1 | `msg_utils.py` | ✅ | msgpack + numpy + JPEG/PNG 序列化 |
| 2 | `policies/base_policy.py` | ✅ | 抽象基类 + RandomPolicy + GoHomePolicy（3阶段） |
| 3 | `policies/act_policy.py` | ✅ | LeRobot ACT 推理 |
| 4 | `policies/pi05_policy.py` | ✅ | Pi0.5 JAX 推理，action chunking |
| 5 | `policies/dp_policy.py` | ✅ | Diffusion Policy PyTorch，关节安全约束 |
| 6 | `camera.py` | ✅ | RealSense 多相机 + DummyCamera |
| 7 | `robot_interface.py` | ✅ | ABC + DummyRobot + KukaIIWA TCP client |
| 8 | `server.py` | ✅ | ZMQ REP 推理服务，`--save` 保存调试数据 |
| 9 | `client.py` | ✅ | ZMQ REQ 本地采集+执行（可选） |
| 10 | `ros_bridge.py` | ✅ | 自包含单文件，相机+ROS+控制循环，iiwa+Robotiq |
| 11 | `config.yaml` | ✅ | 含 server / cameras / robot / control / ros_bridge 配置段 |
| 12 | `test_stage2.py` | ✅ | in-process 集成测试，无硬件端到端验证通过 |

---

## Policy 支持矩阵

| Policy 名 | 文件 | 运行环境 | 特性 |
|-----------|------|----------|------|
| `random` | `base_policy.py` | 任意 | 输出 [-0.05, 0.05] 随机噪声，用于通信验证 |
| `go_home` | `base_policy.py` | 任意 | 3阶段：移到 home → 3次夹爪开合 → 返回初始位 |
| `act` | `act_policy.py` | conda xsim | LeRobot ACT，config.json + model.safetensors |
| `pi05` | `pi05_policy.py` | openpi venv | Pi0.5 JAX LoRA checkpoint，action_horizon=50 |
| `pi05_h50` | `pi05_policy.py` | openpi venv | 同上，但使用不同 config（兼容旧命名） |
| `dp` | `dp_policy.py` | conda xsim | Diffusion Policy PyTorch，含关节安全约束 |

---

## 如何运行

### 阶段 1：无硬件快速测试（DummyRobot + RandomPolicy）

验证 ZMQ 通信链路：

```bash
# 终端 1（主电脑）：启动 server
conda activate xsim
cd /home/rl/projects/X-Sim/real_robot
python server.py --policy random

# 终端 2（任意机器）：启动 dummy bridge
python ros_bridge.py --dummy --server tcp://localhost:5555
```

或使用 in-process 测试脚本（单终端，自动运行 20 步）：

```bash
python test_stage2.py
# 输出：latency、Hz、action shapes
```

### 阶段 2：Policy 推理测试（无真机）

验证模型加载和推理是否正常：

```bash
# Pi0.5
source /home/rl/projects/X-Sim/openpi/.venv/bin/activate
cd /home/rl/projects/X-Sim/real_robot
python server.py --config config.yaml --policy pi05

# 另一个终端
python ros_bridge.py --dummy --server tcp://localhost:5555

# ACT / DP（需要先在 config.yaml 填好 checkpoint 路径）
conda activate xsim
python server.py --config config.yaml --policy act   # 或 dp
python ros_bridge.py --dummy
```

### 阶段 3：真机部署（两台机器）

```
主电脑 (192.168.1.101, GPU)            ROS 机器 (192.168.1.100)
┌────────────────────┐  ZMQ:5555       ┌────────────────────────────┐
│  server.py         │◄───────────────►│  ros_bridge.py             │
│  Pi0.5 / ACT / DP  │                 │  RealSense + iiwa + Robotiq │
└────────────────────┘                 └────────────────────────────┘
```

**步骤 1：** 在 KUKA 平板上启动 ROSSmartServo 应用

**步骤 2：** 把 ros_bridge.py scp 到 ROS 机器（每次修改后重传）：
```bash
scp /home/rl/projects/X-Sim/real_robot/ros_bridge.py jump-server@192.168.1.100:~/
# 如果需要同步 config
scp /home/rl/projects/X-Sim/real_robot/config.yaml jump-server@192.168.1.100:~/
```

**步骤 3：** 在主电脑启动 server：
```bash
# Pi0.5（greycup）
source /home/rl/projects/X-Sim/openpi/.venv/bin/activate
cd /home/rl/projects/X-Sim/real_robot
python server.py --config config.yaml --policy pi05 --save

# Pi0.5（ketchup）
python server.py --config config_ketchup_lite.yaml --policy pi05 --save

# Diffusion Policy
conda activate xsim
python server.py --config config_ketchup_lite.yaml --policy dp --save
```

**步骤 4：** 在 ROS 机器启动 bridge（**必须 source catkin workspace**）：
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
python ros_bridge.py --server tcp://192.168.1.101:5555
```

> **ROS 机器依赖：** `pip install zmq msgpack opencv-python numpy pyrealsense2 pyyaml`
>
> **iiwa_stack：** ROS 机器需编译 `~/catkin_ws/src/iiwa_stack`，提供 `iiwa_msgs`
>
> **建议：** 把 `source ~/catkin_ws/devel/setup.bash` 加入 `~/.bashrc`

---

## 配置文件说明

`config.yaml` 分为五个段：

```yaml
server:
  address: tcp://0.0.0.0:5555
  policy: "dp"                          # random | go_home | act | pi05 | dp
  checkpoint: /path/to/model            # ACT: 目录路径; DP: .pt 文件; Pi05: 见 policies/pi05_policy.py
  device: cuda
  n_action_steps: 8                     # 每次推理执行多少步（buffer 用完再重新推理）
  camera_to_feature_map:
    base_camera: image1
    wrist_camera: image2
  # DP 专用：
  gripper_binarize_threshold: 0.0
  max_joint_delta: 0.15
  xsim_root: /home/rl/projects/X-Sim

cameras:
  - name: base_camera
    serial: "308222301757"              # D455
    width: 640
    height: 480
    fps: 30
    mode: rgb
  - name: wrist_camera
    serial: "306322300452"              # D456
    ...

robot:
  type: kuka_iiwa
  home_position: [0, 0, 0, -1.57, 0, 1.57, 0]
  bridge_host: 192.168.1.100
  bridge_port: 6000

control:
  hz: 15                                # 必须与训练数据帧率匹配（greycup=15Hz）
  max_steps: 800
  num_episodes: 10

ros_bridge:
  server_address: tcp://192.168.1.101:5555
  hz: 15
  jpeg_quality: 50                      # 图像压缩质量（ketchup 用 90）
  max_steps: 800
  num_episodes: 10
```

**关键参数说明：**
- `control.hz`：控制频率，必须与训练数据 fps 匹配（greycup/paper = 15Hz）
- `n_action_steps`：每个 chunk 执行步数，过大导致振荡（15Hz 下建议 ≤8）
- `jpeg_quality`：影响传输带宽和图像质量，通常 50-90

---

## 硬件接口

### ROS Topics（ros_bridge.py）

| 用途 | Topic | 消息类型 | 说明 |
|------|-------|----------|------|
| 关节状态读取 | `/iiwa/state/JointPosition` | `iiwa_msgs/JointPosition` | `position.a1`~`a7`（rad） |
| 关节指令发送 | `/iiwa/command/JointPosition` | `iiwa_msgs/JointPosition` | 同上 |
| 夹爪状态读取 | `/Robotiq2FGripperRobotInput` | `Robotiq2FGripper_robot_input` | `gPO`: 0(开)~255(闭) → 映射 -1~1 |
| 夹爪指令发送 | `/Robotiq2FGripperRobotOutput` | `Robotiq2FGripper_robot_output` | `rPR`: 0(开)~255(闭)，`rSP`=255, `rFR`=150 |

> **注意：** `/iiwa/joint_states` (sensor_msgs/JointState) 虽在 topic list 中，但 iiwa_stack 不在此发布数据，必须用 `/iiwa/state/JointPosition`。

Robotiq 夹爪需要 **5步激活序列**（ros_bridge.py 已实现），激活后方可发送目标位置。

### 相机

| 名称 | 型号 | Serial | 分辨率 |
|------|------|--------|--------|
| base_camera | RealSense D455 | `308222301757` | 640×480 |
| wrist_camera | RealSense D456 | `306322300452` | 640×480 |

> L515 当前不被 pip 版 pyrealsense2 支持，暂不使用。

---

## 通信协议

### ZMQ REQ-REP

```
ros_bridge.py (REQ)                    server.py (REP)
    │──── Observation (msgpack) ──────────►│
    │◄──── Action Chunk (msgpack) ─────────│
    │   (只在 buffer 用完时查询)             │
```

本地 action buffer：bridge 每次查询服务器获取 `n_action_steps` 步的 chunk，本地逐步执行，buffer 耗尽再重新查询。减少网络往返、降低推理频率压力。

### 消息格式

**Observation（Bridge → Server）：**
```python
{
    "images": {
        "base_camera": bytes,       # JPEG 压缩
        "wrist_camera": bytes,      # JPEG 压缩（可选）
    },
    "joint_positions": np.array,    # (7,) float32
    "gripper_state": float,         # -1.0(开) ~ 1.0(闭)，来自 Robotiq gPO
    "timestamp": float,
    "reset": bool                   # episode 重置信号
}
```

**Action Chunk（Server → Bridge）：**
```python
{
    "action": np.array,   # (n_action_steps, 8)：7关节(rad) + 1夹爪(-1~1)
    "done": bool
}
```

---

## Episode 录制与调试

启动 server 时加 `--save` 参数：

```bash
python server.py --config config.yaml --policy pi05 --save
```

每个 episode 保存到 `debug_output/ep{N}/`：
- `states.npy` — (T, 8) 关节+夹爪状态
- `actions.npy` — (T, 8) 发出的动作
- `base_camera.mp4` / `wrist_camera.mp4` — 相机视频
- `query_frames.mp4` — 推理时刻的 base+wrist 并排画面

**对比真机轨迹与训练数据：**
```bash
python plot_states_compare.py
# 生成 debug_output/ep005/state_compare.png
# 8 子图：7 关节 + 1 夹爪，时间轴归一化
```

---

## 添加新 Policy

1. 在 `policies/` 下创建 `your_policy.py`，继承 `BasePolicy`
2. 实现：
   ```python
   class YourPolicy(BasePolicy):
       def load(self, config: dict): ...
       def predict(self, obs: dict) -> np.ndarray:  # 返回 (8,)
           ...
       def reset(self): ...  # 可选
   ```
3. 在 `server.py` 的 `load_policy_from_config()` 加 elif 分支
4. `config.yaml` 设置 `server.policy: "your_policy_name"`

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `No module named 'iiwa_msgs'` | 未 source catkin workspace | `source ~/catkin_ws/devel/setup.bash`（建议加入 `~/.bashrc`） |
| `rostopic echo /iiwa/joint_states` 无输出 | iiwa_stack 不在此 topic 发数据 | 改用 `/iiwa/state/JointPosition` |
| ROS 多机 topic 无数据 | ROS_IP 未设置，publisher 无法回连 subscriber | 检查 `ROS_IP`，确保两机器互通 |
| `joints.csv` 全为 0 | 订阅了错误 topic 或 callback 未触发 | 确认 `/iiwa/state/JointPosition` 有数据 |
| `Device or resource busy`（相机） | 两相机抢同一设备 | 确保 `serial` 字段在 config 中指定 |
| `The Motion is no longer executed` | SmartServo 崩溃（关节跳变过大） | 降低 `max_joint_delta`；重启 ROSSmartServo |
| 机器人振荡 | `n_action_steps` 过大或 `hz` 与训练不匹配 | 设 `n_action_steps ≤ 8`，`hz = 15` |
| Pi0.5 推理慢 / OOM | JAX 显存不足 | `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |
| L515 不被 pyrealsense2 识别 | pip 版本不支持 L515 | 当前仅使用 D455/D456 |
| `--save-images` 无效 | 旧参数名已弃用 | 改用 `--save` |

---

