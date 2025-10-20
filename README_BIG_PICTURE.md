# RC Car Vision-Language Control: Complete Pipeline

> From 3D Gaussian Splatting to Real-World Deployment

**Goal**: Train an RC car to navigate your room using natural language commands and onboard camera, with policies learned entirely in simulation and transferred to real hardware.

---

## 🎯 Project Vision

**Input**: "Go to the kitchen" + Camera feed  
**Output**: Steering angle + Throttle commands  
**Hardware**: Custom RC car with Jetson Orin Nano + MCU for motor control  
**Training**: Isaac Lab simulation in photorealistic 3D Gaussian Splatting reconstruction

---

## 📋 Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: ENVIRONMENT                      │
│  3D Scan Room → Gaussian Splatting → Mesh → Isaac Sim       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2: ROBOT MODEL                      │
│  CAD/Measure RC Car → URDF/USD → Isaac Lab Asset            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3: SIMULATION                       │
│  Camera Sensor → Physics → Teleoperation → Data Collection  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 4: TRAINING                         │
│  Vision-Language Policy → Imitation/RL → Domain Random.     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 5: DEPLOYMENT                       │
│  TensorRT Optimization → Jetson → MCU → Real Car            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
RC-Sim2Real/
├── README.md                           # Main project overview
├── README_BIG_PICTURE.md              # This file - detailed pipeline
├── docs/
│   ├── phase1_environment.md          # 3D scanning guide
│   ├── phase2_robot_model.md          # RC car modeling
│   ├── phase3_simulation.md           # Simulation setup
│   ├── phase4_training.md             # Training strategies
│   └── phase5_deployment.md           # Real-world deployment
│
├── isaac_lab/                         # Isaac Lab fork (separate repo)
│   └── source/
│       └── isaaclab_assets/
│           └── isaaclab_assets/
│               ├── robots/
│               │   └── rc_car.py      # Your RC car configuration
│               └── sensors/
│                   └── rc_camera.py   # Camera sensor config
│
├── environments/                      # Custom environments
│   ├── __init__.py
│   ├── room_env.py                   # Base room environment
│   ├── room_env_cfg.py               # Environment configuration
│   └── tasks/
│       ├── language_nav_task.py      # Vision-language navigation
│       └── language_nav_task_cfg.py
│
├── models/                           # Neural network architectures
│   ├── __init__.py
│   ├── vision_language_policy.py    # Main policy network
│   ├── clip_encoder.py              # Vision encoder (CLIP-based)
│   └── language_encoder.py          # Language encoder
│
├── training/                         # Training scripts
│   ├── __init__.py
│   ├── train_bc.py                  # Behavior cloning (imitation)
│   ├── train_ppo.py                 # PPO reinforcement learning
│   ├── train_dagger.py              # DAgger (interactive imitation)
│   └── config/
│       ├── bc_config.yaml
│       ├── ppo_config.yaml
│       └── domain_randomization.yaml
│
├── data_collection/                  # Teleoperation & data
│   ├── __init__.py
│   ├── teleop_keyboard.py           # Keyboard control
│   ├── teleop_joystick.py           # Joystick/gamepad control
│   ├── record_demonstrations.py      # Save trajectories
│   └── dataset/                      # Collected demonstrations
│       ├── episode_000/
│       ├── episode_001/
│       └── ...
│
├── deployment/                       # Real hardware deployment
│   ├── jetson/
│   │   ├── inference_node.py        # Main inference loop
│   │   ├── camera_driver.py         # Camera interface
│   │   ├── mcu_interface.py         # Serial comm to MCU
│   │   └── requirements.txt
│   ├── mcu/
│   │   ├── rc_car_controller/       # Arduino/PlatformIO project
│   │   │   ├── src/
│   │   │   │   └── main.cpp        # PWM control firmware
│   │   │   └── platformio.ini
│   │   └── protocol.md              # Communication protocol
│   └── optimize/
│       ├── export_onnx.py           # PyTorch → ONNX
│       └── build_tensorrt.py        # ONNX → TensorRT
│
├── assets/                          # 3D models and scans
│   ├── room_scan/
│   │   ├── images/                  # Input photos for NeRF/3DGS
│   │   ├── gaussian_splat.ply       # 3DGS output
│   │   ├── room_mesh.obj            # Converted mesh
│   │   └── room.usd                 # Final USD for Isaac Sim
│   └── rc_car/
│       ├── rc_car.urdf              # Robot description
│       ├── rc_car.usd               # USD version
│       └── meshes/                  # Visual and collision meshes
│
├── scripts/                         # Utility scripts
│   ├── convert_splat_to_mesh.py    # 3DGS → mesh conversion
│   ├── mesh_to_usd.py              # OBJ → USD conversion
│   ├── calibrate_camera.py         # Camera intrinsics
│   └── domain_randomization_test.py
│
├── tests/                           # Unit and integration tests
│   ├── test_environment.py
│   ├── test_policy.py
│   └── test_deployment.py
│
├── configs/                         # Global configurations
│   ├── hardware_specs.yaml         # Real RC car specifications
│   ├── camera_calibration.yaml     # Camera parameters
│   └── simulation_params.yaml      # Physics parameters
│
└── notebooks/                       # Jupyter notebooks for analysis
    ├── visualize_trajectories.ipynb
    ├── analyze_training.ipynb
    └── sim2real_comparison.ipynb
```

---

## 📊 Phase Breakdown

### **PHASE 1: Environment Reconstruction** (Week 1)

**Goal**: Create a photorealistic digital twin of your room in Isaac Sim

#### 1.1 Room Scanning
- **Tool**: Smartphone camera or DSLR
- **Process**:
  - Capture 100-200 photos of your room
  - Cover all angles, especially floor where car will drive
  - Ensure good lighting and overlap between images
  - Take photos at car's eye level (~10-20cm from ground)

#### 1.2 3D Gaussian Splatting
- **Tool**: Nerfstudio
- **Commands**:
  ```bash
  # Install Nerfstudio
  pip install nerfstudio
  
  # Process images
  ns-process-data images --data assets/room_scan/images --output-dir assets/room_scan/processed
  
  # Train Gaussian Splatting
  ns-train splatfacto --data assets/room_scan/processed
  
  # Export
  ns-export gaussian-splat --load-config outputs/room_scan/splatfacto/*/config.yml --output-dir assets/room_scan/
  ```

#### 1.3 Mesh Conversion
- **Why**: Isaac Sim needs collision meshes for physics
- **Process**:
  ```bash
  # Run conversion script (to be created)
  python scripts/convert_splat_to_mesh.py \
    --input assets/room_scan/gaussian_splat.ply \
    --output assets/room_scan/room_mesh.obj \
    --voxel-size 0.05
  ```

#### 1.4 USD Conversion
- **Tool**: Omniverse USD Composer or Python script
- **Process**:
  ```bash
  python scripts/mesh_to_usd.py \
    --input assets/room_scan/room_mesh.obj \
    --output assets/room_scan/room.usd \
    --add-collision \
    --material floor
  ```

**Deliverables**:
- ✅ `assets/room_scan/room.usd` - Ready for Isaac Sim
- ✅ Collision meshes with proper physics materials
- ✅ Lighting that matches real room

---

### **PHASE 2: RC Car Modeling** (Week 1-2)

**Goal**: Create accurate digital model of your RC car

#### 2.1 Measurements & Specifications
- **Document** (in `configs/hardware_specs.yaml`):
  ```yaml
  rc_car:
    # Dimensions (meters)
    length: 0.35
    width: 0.20
    height: 0.12
    wheelbase: 0.25
    track_width: 0.18
    wheel_radius: 0.04
    
    # Mass properties (kg)
    total_mass: 1.2
    chassis_mass: 0.8
    wheel_mass: 0.1
    
    # Motor specs
    max_steering_angle: 30  # degrees
    max_speed: 5.0  # m/s
    max_acceleration: 3.0  # m/s^2
    
    # Sensors
    camera:
      resolution: [640, 480]
      fov: 90  # degrees
      fps: 30
      position: [0.15, 0.0, 0.08]  # x, y, z relative to chassis
      orientation: [0, 15, 0]  # roll, pitch, yaw (degrees)
  ```

#### 2.2 URDF Creation
- **Location**: `assets/rc_car/rc_car.urdf`
- **Structure**:
  ```xml
  <!-- Base chassis -->
  <link name="base_link">
    <visual>...</visual>
    <collision>...</collision>
    <inertial>...</inertial>
  </link>
  
  <!-- Wheels with joints -->
  <!-- Front wheels: revolute for steering -->
  <!-- Rear wheels: continuous for drive -->
  
  <!-- Camera mount -->
  <link name="camera_link">...</link>
  ```

#### 2.3 USD Conversion for Isaac Sim
- **Convert URDF to USD**:
  ```bash
  # Isaac Lab provides conversion tools
  cd isaac_lab
  ./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
    ../assets/rc_car/rc_car.urdf \
    ../assets/rc_car/rc_car.usd
  ```

#### 2.4 Isaac Lab Asset Configuration
- **File**: `isaac_lab/source/isaaclab_assets/isaaclab_assets/robots/rc_car.py`
- **Template**:
  ```python
  from isaaclab.assets import ArticulationCfg
  from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
  
  RC_CAR_CFG = ArticulationCfg(
      prim_path="{ENV_REGEX_NS}/Robot",
      spawn=sim_utils.UsdFileCfg(
          usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/RCCar/rc_car.usd",
          rigid_props=sim_utils.RigidBodyPropertiesCfg(
              rigid_body_enabled=True,
              max_linear_velocity=5.0,
              max_angular_velocity=10.0,
          ),
          articulation_props=sim_utils.ArticulationRootPropertiesCfg(
              enabled_self_collisions=False,
          ),
      ),
      init_state=ArticulationCfg.InitialStateCfg(
          pos=(0.0, 0.0, 0.1),
          joint_pos={
              "front_left_steering": 0.0,
              "front_right_steering": 0.0,
              "rear_left_wheel": 0.0,
              "rear_right_wheel": 0.0,
          },
      ),
      actuators={
          "steering": ImplicitActuatorCfg(
              joint_names_expr=[".*steering"],
              stiffness=200.0,
              damping=20.0,
          ),
          "drive": ImplicitActuatorCfg(
              joint_names_expr=[".*wheel"],
              stiffness=0.0,
              damping=10.0,
          ),
      },
  )
  ```

**Deliverables**:
- ✅ `assets/rc_car/rc_car.usd` - Ready for Isaac Sim
- ✅ `robots/rc_car.py` - Isaac Lab configuration
- ✅ `configs/hardware_specs.yaml` - Complete specifications
- ✅ Validated physics in simulation

---

### **PHASE 3: Simulation Environment** (Week 2-3)

**Goal**: Create interactive training environment in Isaac Lab

#### 3.1 Base Environment Setup
- **File**: `environments/room_env.py`
- **Components**:
  ```python
  from isaaclab.envs import DirectRLEnv
  from isaaclab.scene import InteractiveSceneCfg
  from isaaclab.sensors import CameraCfg
  
  class RoomNavEnv(DirectRLEnv):
      def __init__(self, cfg: RoomNavEnvCfg):
          super().__init__(cfg)
          
          # Scene: Room + RC Car + Camera
          # Observations: Camera images + language embeddings
          # Actions: [steering, throttle]
          # Rewards: Progress toward goal, collision penalty
  ```

#### 3.2 Camera Sensor Configuration
- **File**: `isaac_lab/source/isaaclab_assets/isaaclab_assets/sensors/rc_camera.py`
- **Match real camera specs**:
  ```python
  from isaaclab.sensors import CameraCfg
  
  RC_CAMERA_CFG = CameraCfg(
      prim_path="{ENV_REGEX_NS}/Robot/camera_link/camera",
      update_period=0.033,  # 30 FPS
      height=480,
      width=640,
      data_types=["rgb"],
      spawn=sim_utils.PinholeCameraCfg(
          focal_length=3.0,  # mm, from calibration
          horizontal_aperture=5.76,  # mm
          clipping_range=(0.01, 10.0),
      ),
  )
  ```

#### 3.3 Teleoperation Interface
- **File**: `data_collection/teleop_keyboard.py`
- **Features**:
  - Arrow keys for steering/throttle
  - Language command input
  - Record button to save demonstrations
  - Real-time visualization

- **Example**:
  ```python
  import pygame
  from isaaclab.app import AppLauncher
  
  class TeleopInterface:
      def __init__(self, env):
          self.env = env
          pygame.init()
          self.screen = pygame.display.set_mode((800, 600))
          
      def run(self):
          recording = False
          episode_data = []
          
          while True:
              keys = pygame.key.get_pressed()
              
              # Map keys to actions
              steering = 0.0
              throttle = 0.0
              if keys[pygame.K_LEFT]: steering = -1.0
              if keys[pygame.K_RIGHT]: steering = 1.0
              if keys[pygame.K_UP]: throttle = 1.0
              if keys[pygame.K_DOWN]: throttle = -1.0
              
              # Record if 'R' pressed
              if keys[pygame.K_r] and not recording:
                  recording = True
                  print("Recording started")
              
              # Apply action
              obs, reward, done, info = self.env.step([steering, throttle])
              
              if recording:
                  episode_data.append({
                      'image': obs['camera'],
                      'instruction': self.current_instruction,
                      'action': [steering, throttle]
                  })
  ```

#### 3.4 Domain Randomization
- **File**: `training/config/domain_randomization.yaml`
- **Parameters to randomize**:
  ```yaml
  domain_randomization:
    camera:
      brightness: [0.8, 1.2]
      contrast: [0.8, 1.2]
      saturation: [0.8, 1.2]
      hue: [-0.1, 0.1]
      gaussian_noise_std: [0.0, 0.02]
      motion_blur_kernel: [0, 5]
    
    lighting:
      intensity: [0.5, 2.0]
      color_temp: [3000, 7000]  # Kelvin
      position_offset: [-0.5, 0.5]
    
    physics:
      wheel_friction: [0.6, 1.2]
      floor_friction: [0.5, 1.0]
      mass_scale: [0.9, 1.1]
      motor_latency: [0.0, 0.05]  # seconds
      action_noise: [0.0, 0.05]
    
    textures:
      floor_texture_set: ['wood', 'tile', 'carpet', 'concrete']
      randomize_every_n_steps: 100
  ```

**Deliverables**:
- ✅ `environments/room_env.py` - Fully functional environment
- ✅ Teleoperation interface working
- ✅ Domain randomization implemented
- ✅ Data recording pipeline ready

---

### **PHASE 4: Policy Training** (Week 3-5)

**Goal**: Train vision-language policy for navigation

#### 4.1 Vision-Language Architecture
- **File**: `models/vision_language_policy.py`
- **Architecture**:
  ```python
  import torch
  import torch.nn as nn
  from transformers import CLIPVisionModel, CLIPTextModel
  
  class VisionLanguagePolicy(nn.Module):
      def __init__(self):
          super().__init__()
          
          # Vision encoder (pretrained CLIP)
          self.vision_encoder = CLIPVisionModel.from_pretrained(
              "openai/clip-vit-base-patch32"
          )
          
          # Language encoder (pretrained CLIP)
          self.language_encoder = CLIPTextModel.from_pretrained(
              "openai/clip-vit-base-patch32"
          )
          
          # Fusion layer
          self.fusion = nn.Sequential(
              nn.Linear(512 + 512, 512),  # CLIP embeddings are 512-dim
              nn.ReLU(),
              nn.Dropout(0.1),
          )
          
          # Policy head
          self.actor = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 2),  # [steering, throttle]
              nn.Tanh(),  # Output in [-1, 1]
          )
          
          # Value head (for RL)
          self.critic = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 1),
          )
      
      def forward(self, image, text):
          # Encode image
          vision_features = self.vision_encoder(pixel_values=image).pooler_output
          
          # Encode text
          language_features = self.language_encoder(input_ids=text).pooler_output
          
          # Fuse
          fused = torch.cat([vision_features, language_features], dim=-1)
          fused = self.fusion(fused)
          
          # Policy and value
          action = self.actor(fused)
          value = self.critic(fused)
          
          return action, value
  ```

#### 4.2 Training Strategy

**Option A: Behavior Cloning (Start Here)**
- **File**: `training/train_bc.py`
- **Process**:
  1. Collect 50-100 teleoperation demonstrations
  2. Train supervised on human demonstrations
  3. Fast iteration, good baseline

**Option B: DAgger (Interactive Imitation)**
- **File**: `training/train_dagger.py`
- **Process**:
  1. Train initial policy on demonstrations
  2. Deploy policy in simulation
  3. When policy fails, human corrects
  4. Add corrections to dataset
  5. Retrain
  6. Iterate

**Option C: Reinforcement Learning (PPO)**
- **File**: `training/train_ppo.py`
- **Process**:
  1. Initialize with BC policy
  2. Fine-tune with PPO
  3. Reward function:
     ```python
     reward = (
         progress_to_goal * 10.0
         - collision_penalty * 5.0
         - smooth_driving_bonus * 0.1
         - instruction_alignment * 2.0
     )
     ```

#### 4.3 Training Curriculum
- **Week 3**: Collect demonstrations, train BC
- **Week 4**: Train PPO with simple instructions
- **Week 5**: Add complex instructions, domain randomization

#### 4.4 Evaluation Metrics
```yaml
metrics:
  success_rate: >0.8  # Reaches goal
  collision_rate: <0.1  # Doesn't crash
  avg_episode_length: <100 steps  # Efficient
  instruction_following: >0.85  # Semantic alignment
```

**Deliverables**:
- ✅ Trained BC policy (baseline)
- ✅ Trained PPO policy (optimized)
- ✅ Evaluation benchmarks
- ✅ Training logs and visualizations

---

### **PHASE 5: Deployment** (Week 5-6)

**Goal**: Deploy trained policy to real RC car

#### 5.1 Model Optimization
- **File**: `deployment/optimize/export_onnx.py`
- **Process**:
  ```python
  import torch
  import torch.onnx
  
  # Load trained policy
  policy = VisionLanguagePolicy()
  policy.load_state_dict(torch.load('checkpoints/best_policy.pth'))
  policy.eval()
  
  # Dummy inputs
  dummy_image = torch.randn(1, 3, 480, 640)
  dummy_text = torch.randint(0, 50000, (1, 77))  # CLIP text tokens
  
  # Export to ONNX
  torch.onnx.export(
      policy,
      (dummy_image, dummy_text),
      'deployment/optimize/policy.onnx',
      input_names=['image', 'text'],
      output_names=['action'],
      dynamic_axes={
          'image': {0: 'batch'},
          'text': {0: 'batch'},
          'action': {0: 'batch'}
      }
  )
  ```

- **File**: `deployment/optimize/build_tensorrt.py`
- **Process**:
  ```bash
  # Convert ONNX to TensorRT
  /usr/src/tensorrt/bin/trtexec \
    --onnx=policy.onnx \
    --saveEngine=policy.trt \
    --fp16 \
    --workspace=4096 \
    --verbose
  ```

#### 5.2 Jetson Inference Node
- **File**: `deployment/jetson/inference_node.py`
- **Main loop**:
  ```python
  import cv2
  import numpy as np
  import tensorrt as trt
  import pycuda.driver as cuda
  from transformers import CLIPTokenizer
  from mcu_interface import MCUInterface
  
  class InferenceNode:
      def __init__(self):
          # Load TensorRT engine
          self.engine = self.load_engine('policy.trt')
          self.context = self.engine.create_execution_context()
          
          # Camera
          self.camera = cv2.VideoCapture(0)
          self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
          self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
          self.camera.set(cv2.CAP_PROP_FPS, 30)
          
          # Text tokenizer
          self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
          
          # MCU interface
          self.mcu = MCUInterface('/dev/ttyTHS1', 115200)
          
          # Current instruction
          self.instruction = "explore the room"
      
      def run(self):
          while True:
              # Capture frame
              ret, frame = self.camera.read()
              if not ret:
                  continue
              
              # Preprocess image
              image = self.preprocess_image(frame)
              
              # Tokenize instruction
              text_tokens = self.tokenizer(
                  self.instruction,
                  padding='max_length',
                  max_length=77,
                  return_tensors='np'
              )['input_ids']
              
              # Inference
              action = self.infer(image, text_tokens)
              steering, throttle = action[0], action[1]
              
              # Send to MCU
              self.mcu.send_command(steering, throttle)
              
              # Display (optional)
              cv2.imshow('RC Car Vision', frame)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
      
      def preprocess_image(self, frame):
          # Resize, normalize, etc.
          img = cv2.resize(frame, (640, 480))
          img = img.astype(np.float32) / 255.0
          img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
          img = np.transpose(img, (2, 0, 1))
          return img[np.newaxis, :]
  ```

#### 5.3 MCU Firmware
- **File**: `deployment/mcu/rc_car_controller/src/main.cpp`
- **Arduino-style code**:
  ```cpp
  #include <Arduino.h>
  #include <Servo.h>
  
  // Hardware setup
  Servo steering_servo;
  Servo esc;  // Electronic Speed Controller
  
  const int STEERING_PIN = 9;
  const int ESC_PIN = 10;
  
  // Command protocol
  struct Command {
      float steering;  // -1.0 to 1.0
      float throttle;  // -1.0 to 1.0
  };
  
  Command current_cmd = {0.0, 0.0};
  
  void setup() {
      Serial.begin(115200);
      
      steering_servo.attach(STEERING_PIN);
      esc.attach(ESC_PIN);
      
      // Initialize to neutral
      steering_servo.writeMicroseconds(1500);
      esc.writeMicroseconds(1500);
      
      // ESC calibration sequence
      delay(1000);
      esc.writeMicroseconds(1500);
      delay(1000);
  }
  
  void loop() {
      // Read commands from Jetson
      if (Serial.available() >= sizeof(Command)) {
          Serial.readBytes((char*)&current_cmd, sizeof(Command));
          
          // Convert to PWM (1000-2000 microseconds)
          int steering_pwm = 1500 + (int)(current_cmd.steering * 500);
          int throttle_pwm = 1500 + (int)(current_cmd.throttle * 500);
          
          // Clamp values
          steering_pwm = constrain(steering_pwm, 1000, 2000);
          throttle_pwm = constrain(throttle_pwm, 1000, 2000);
          
          // Apply
          steering_servo.writeMicroseconds(steering_pwm);
          esc.writeMicroseconds(throttle_pwm);
      }
      
      delay(10);  // 100Hz control loop
  }
  ```

#### 5.4 Communication Protocol
- **File**: `deployment/mcu/protocol.md`
- **Binary protocol**:
  ```
  Jetson → MCU (8 bytes per packet):
  [Header: 0xFF][Steering: float32][Throttle: float32]
  
  MCU → Jetson (4 bytes per packet):
  [Header: 0xFF][Status: uint8][Battery: uint8][Reserved: uint8]
  
  Status bits:
  - Bit 0: Motor enabled
  - Bit 1: Low battery
  - Bit 2: Error state
  ```

#### 5.5 Deployment Checklist
```markdown
- [ ] Hardware assembly
  - [ ] Jetson Orin Nano mounted
  - [ ] Camera positioned correctly (matches simulation)
  - [ ] MCU connected via UART
  - [ ] Power supply adequate (battery capacity)
  
- [ ] Software setup
  - [ ] TensorRT engine copied to Jetson
  - [ ] MCU firmware flashed
  - [ ] Camera calibration done
  - [ ] Serial communication tested
  
- [ ] Testing
  - [ ] Manual control via MCU (no Jetson)
  - [ ] Camera feed working
  - [ ] Inference latency <50ms
  - [ ] End-to-end latency <100ms
  
- [ ] Sim-to-real validation
  - [ ] Test in real room (same as scan)
  - [ ] Compare trajectories to simulation
  - [ ] Measure success rate
  - [ ] Identify failure modes
```

**Deliverables**:
- ✅ TensorRT optimized model running on Jetson
- ✅ MCU firmware controlling motors
- ✅ End-to-end system latency <100ms
- ✅ Real-world validation in room

---

## 🔬 Advanced Features (Post-MVP)

### Semantic Mapping
- Build semantic map of room during exploration
- Use for better language grounding ("go to the table" → table location)

### Multi-task Learning
- Train on multiple instructions simultaneously
- "Pick up the object" + manipulation arm

### Active Learning
- Detect when policy is uncertain
- Request human intervention/correction
- Continuous improvement

### Hierarchical Control
- High-level planner: Language → waypoints
- Low-level controller: Waypoints → steering/throttle

### Real-time Adaptation
- Online learning from real-world experience
- Adapt to new rooms without retraining

---

## 📈 Success Metrics

### Simulation
- ✅ Success rate >80% on diverse instructions
- ✅ Collision rate <10%
- ✅ Smooth trajectories (low jerk)

### Real World
- ✅ Success rate >70% (sim-to-real gap)
- ✅ Safe operation (no crashes)
- ✅ Natural language understanding (diverse commands)
- ✅ Latency <100ms (end-to-end)

---

## 🛠️ Development Workflow

### Daily Development
```bash
# Activate environment
cd /home/maxboels/projects/RC-Sim2Real
source env_isaaclab/bin/activate

# Run simulation
cd isaac_lab
./isaaclab.sh -p ../environments/room_env.py

# Train policy
python ../training/train_bc.py --config ../training/config/bc_config.yaml

# Test on Jetson (when ready)
ssh jetson@192.168.1.100
python3 deployment/jetson/inference_node.py
```

### Git Workflow
```bash
# Feature branch for each phase
git checkout -b phase1/environment-scan
# ... work ...
git commit -m "Complete room 3D scan and USD conversion"
git push origin phase1/environment-scan

# Isaac Lab fork
cd isaac_lab
git checkout -b feature/rc-car-robot
# ... add rc_car.py ...
git commit -m "Add RC car robot configuration"
git push origin feature/rc-car-robot
```

---

## 📚 Key Resources

### Documentation
- [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Nerfstudio Docs](https://docs.nerf.studio/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

### Papers & References
- **3D Gaussian Splatting**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- **Vision-Language Navigation**: "RT-2: Vision-Language-Action Models"
- **Sim-to-Real**: "Learning Dexterous In-Hand Manipulation" (OpenAI)
- **Domain Randomization**: "Domain Randomization for Transferring Deep Neural Networks"

### Code Examples
- Isaac Lab: `source/standalone/tutorials/`
- Isaac Lab Environments: `source/isaaclab_tasks/`
- Isaac Lab RL: `source/isaaclab_rl/`

---

## ⚠️ Common Pitfalls & Solutions

### Problem: Gaussian Splats don't work in physics sim
**Solution**: Extract mesh for collision, optionally keep splats for visual rendering

### Problem: Sim-to-real gap too large
**Solution**: 
- Heavy domain randomization
- Match camera parameters exactly
- Calibrate physics parameters from real car

### Problem: Policy doesn't generalize to new instructions
**Solution**:
- Collect diverse training instructions
- Use pretrained language encoders (CLIP)
- Train with instruction augmentation

### Problem: Real-world latency too high
**Solution**:
- Optimize with TensorRT FP16
- Reduce image resolution if needed
- Use asynchronous inference

### Problem: Safety concerns in real deployment
**Solution**:
- Start with low speeds
- Add emergency stop (physical button)
- Geofence the car (virtual boundaries)
- Monitor battery level

---

## 🎯 Milestones & Timeline

### Week 1: Foundation
- [x] Project setup
- [ ] Room scan complete
- [ ] USD environment loaded in Isaac Sim
- [ ] RC car measurements documented

### Week 2: Simulation
- [ ] RC car model in Isaac Lab
- [ ] Camera sensor working
- [ ] Teleoperation functional
- [ ] 10 demonstration episodes collected

### Week 3: Training
- [ ] BC policy trained
- [ ] Evaluation metrics implemented
- [ ] 50 demonstration episodes collected
- [ ] Domain randomization working

### Week 4: Advanced Training
- [ ] PPO training running
- [ ] Policy achieves >80% success in sim
- [ ] Complex instructions tested

### Week 5: Deployment Prep
- [ ] Model optimized (TensorRT)
- [ ] MCU firmware tested
- [ ] Jetson inference node ready

### Week 6: Real World
- [ ] Hardware assembled
- [ ] End-to-end system tested
- [ ] Real-world validation
- [ ] Documentation complete

---

## 🤝 Contributing

This is a personal research project, but contributions to improve the pipeline are welcome!

### How to Contribute
1. Fork the main repo
2. Create feature branch
3. Make improvements
4. Submit pull request

### Areas for Contribution
- Better domain randomization techniques
- More efficient policy architectures
- Improved sim-to-real transfer
- Additional sensors (LiDAR, depth camera)

---

## 📝 Notes

- **Safety First**: Always test in controlled environment
- **Iterate Fast**: Start simple, add complexity gradually
- **Document Everything**: Keep detailed notes on parameters, experiments
- **Version Control**: Commit often, especially before major changes
- **Backup Models**: Save checkpoints frequently

---

**Last Updated**: October 20, 2025  
**Next Milestone**: Phase 1 - Room Scanning  
**Current Status**: 🚀 Project Initialized
