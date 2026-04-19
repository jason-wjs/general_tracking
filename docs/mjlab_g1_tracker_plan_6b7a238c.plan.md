---
name: general_tracking
overview: 在 mjlab 之上构建下游插件包 general_tracking，语义忠实复现 ProtoMotions examples/experiments/mimic/mlp_bm_l2c2.py 的 G1 general motion tracker 训练管线（7 项 reward 含 global_anchor_ori + 4 个 region-weighted body 项、noisy/clean reduced-coords obs + max-coords critic、future_steps=[1,2,4,8]、L2C2 lambda=1.0、双 optimizer、MimicEvaluator 风格的周期性 motion reweighting）。消费用户已有 lafan1_npz 的 mjlab NPZ 格式，time 索引走 Option A（整数 step），`control_fps=50 Hz` 由 manifest 声明并与 env control dt 强校验。扩展 rsl-rl PPO 而非完全自写。v1 单 GPU；v2 多 GPU 分片。
todos:
  - id: bootstrap-package
    content: 建包骨架 general_tracking（pyproject.toml 含 [project.scripts] 注册 gt-train/gt-play/gt-build-manifest、src/general_tracking、tests、scripts/、LICENSE、README、AGENTS.md，默认依赖 PyPI mjlab，只有确需未发布 upstream 特性时才切 Git pin，注册 mjlab.tasks entry point）
    status: pending
  - id: freeze-body-schema
    content: Phase 1 冻结数据契约（robots/g1/schema.py）：anchor_body_name='torso_link'、完整 body_names 列表（按 mjlab G1 MJCF 顺序）、num_dofs、joint_names、NPZ 字段清单；`control_fps=50` 由 manifest 声明并作为 env/library 一致性不变量
    status: pending
  - id: motion-library-core
    content: MotionLibrary（Option A 整数 step）：多 NPZ 扁平拼接、clip_starts/lengths/weights、sample_clip_ids（multinomial）、sample_init_time（init_start_prob=0.2）、get_state_at、get_future_states（offsets=[1,2,4,8] 末尾 clamp）；加载 manifest 后 assert manifest.control_fps==env control fps
    status: pending
  - id: manifest-builder
    content: src/general_tracking/data/cli/build_manifest.py（Python CLI）：扫目录、读取每个 NPZ 的 frame 数与路径，写出 motion_manifest.yaml；`control_fps` 由 CLI 参数 `--control-fps` 注入（默认 50.0）；通过 pyproject [project.scripts] 暴露为 gt-build-manifest
    status: pending
  - id: shell-launchers
    content: scripts/train.sh、scripts/play.sh、scripts/build_manifest.sh：薄 wrapper，调用 uv run gt-train/gt-play/gt-build-manifest；支持环境变量（NUM_ENVS、MAX_ITER、CKPT、MOTION_LIB_PATH）
    status: pending
  - id: multi-clip-command
    content: MultiClipMotionCommand(Cfg)：clip_ids+time_steps 双 tensor、_resample_command 写 root+joint+重置 noise、_update_command 推进、_update_metrics、anchor_quat_w/body_pos_w 等 @property、future 多步查询
    status: pending
  - id: reduced-coords-obs
    content: reduced_coords_obs 函数（dof_pos, dof_vel, root_local_ang_vel, projected_gravity_b(anchor)）；对应 ProtoMotions reduced_coords_obs_factory(root_height_obs=False, root_vel_obs=False)
    status: pending
  - id: reduced-coords-target-poses
    content: reduced_coords_target_poses 函数（future_steps=[1,2,4,8]，每步输出 rel_anchor_rot_6d(6) + target_dof_vel(nd) + target_dof_pos(nd)；include_xy_offset=False/include_height=False/include_anchor_vel=False/include_anchor_ang_vel=False）
    status: pending
  - id: max-coords-obs-critic
    content: max_coords_obs 函数（body_pos/quat/vel/ang_vel 全部在 heading-aligned local 坐标系 + root_height, local_obs=True, observe_contacts=False）
    status: pending
  - id: max-coords-target-poses-critic
    content: max_coords_target_poses 函数（future_steps=[1,2,4,8]，with_velocities=True, with_relative=True）
    status: pending
  - id: processed-action-history
    content: processed_action_history 观察项（history_steps=1, processed=True：仅上一步 processed action；非多步 history）；明确由自定义 BMPositionAction 或薄 env/action wrapper 维护 1-slot buffer，reset 清零
    status: pending
  - id: clean-noisy-obs-groups
    content: 3 组 observation group：actor_noisy（reduced_coords_obs + reduced_coords_target_poses + processed_action_history，挂 Unoise）、actor_clean（同源无噪声）、critic（max_coords_obs + max_coords_target_poses + processed_action_history，无噪声）
    status: pending
  - id: global-anchor-ori-reward
    content: 复用 mjlab motion_global_anchor_orientation_error_exp 作为 'motion_global_anchor_ori'（weight=0.5, std=0.4）
    status: pending
  - id: region-density-weights
    content: 在 robots/g1/schema.py 里实现 compute_body_density_weights（chain-distance density, discount=0.9, 归一化到 sum=num_bodies），返回 cached tensor 供 reward 引用；对齐 ProtoMotions pose_lib.compute_body_density_weights
    status: pending
  - id: region-weighted-reward-kernels
    content: 4 个 region-weighted reward kernel（region_weighted_body_position_error_exp/orientation/linear_velocity/angular_velocity），按 density_weights per-body 加权后对 body 维求均值；对齐 ProtoMotions use_region_weights=True
    status: pending
  - id: reward-set-assembly
    content: 组装 7 项 reward（global_anchor_ori=0.5, rel_body_pos=1.0 σ=0.3, rel_body_ori=1.0 σ=0.4, body_lin_vel=1.0 σ=1.0, body_ang_vel=1.0 σ=3.14 全部 region-weighted, action_rate=-0.1, joint_pos_limits=-10.0）；不含 anchor_pos / self_collision / contact
    status: pending
  - id: single-fall-termination
    content: 唯一 fall termination：复用 mjlab motion_anchor_height_error termination 并把 threshold 固定为 0.25；去掉 anchor_ori / body_pos_z / undesired_contacts
    status: pending
  - id: bm-pd-action
    content: BMPositionAction（subclass mjlab JointPositionAction）：复用父类 process_actions 数学（`_processed = _raw * _scale + _offset`）；scale dict 由 `build_g1_bm_action_scale()` 从 `G1_ARTICULATION.actuators` 推 `effort_limit/stiffness`（对齐 ProtoMotions `action_functions.py:403`，不是 mjlab `G1_ACTION_SCALE` 的 0.25*e/s）；use_default_offset=True；额外维护 1-slot `_history` 供 processed_action_history observation 读取，reset 清零；补充 `robots/g1/action_scale.py`
    status: pending
  - id: dr-events
    content: DR events：复用 mjlab 的 geom_friction/body_com_offset/push_by_setting_velocity/encoder_bias；新增 action_noise（每 step 给 processed_actions 加 Unoise）和 reset_noise（reset 时给 qpos/qvel 加 Unoise）
    status: pending
  - id: actor-critic-network
    content: GeneralTrackingActorCritic：actor MLP 6x1024 ReLU、critic MLP 4x1024 ReLU、分离参数（无共享 backbone）、init_logstd=-2.9、learnable_std=True、empirical_normalization 打开（等价 normalize_obs + clamp=5）
    status: pending
  - id: l2c2-loss-module
    content: learning/ppo/l2c2.py：compute_l2c2_loss(mu_noisy, mu_clean, obs_pairs, lambda_coef)：对齐 ProtoMotions `agent.py:497-524`，`input_ss = Σ diff.pow(2).sum()` / `input_n = Σ diff.numel()`、`input_dist = (input_ss / input_n).detach()`（detach 避免反向鼓励 obs pipeline 放大 input 差）、`output_dist = (mu_noisy - mu_clean).pow(2).mean()`（scalar），返回 `lambda * output_dist / (input_dist + 1e-8)`；单测覆盖 ProtoMotions 同公式、lambda=0 等价 baseline、input_dist detach 不反向
    status: pending
  - id: general-tracking-ppo
    content: GeneralTrackingPPO(rsl_rl.PPO)：双 optimizer（actor lr=2e-5, critic lr=1e-4, betas=(0.95,0.99)）、override update() 增加 actor_clean forward + L2C2 loss（lambda_l2c2=1.0）、num_learning_epochs=2、num_mini_batches=4、gradient_clip_val=50、clip_critic_loss=True、normalize_rewards=False、advantage_normalization(shift_mean=True)、adaptive_lr=False、entropy_coef=0
    status: pending
  - id: motion-success-evaluator
    content: MotionSuccessEvaluator（ProtoMotions MimicEvaluator 风格周期扫描）：每 eval_metrics_every=200 epoch 暂停训练→系统扫描全 clip（每 env 从 motion_times=0 播放到 max_eval_steps=600 或 clip 终止）→记录 6 个 evaluation metrics 作训练日志（anchor_ori / relative_body_pos / anchor_height_error / gt_error / gr_error / max_joint_error），**fail/reweight 仅由 `anchor_height_error > 0.25` 触发**（对齐 ProtoMotions `utils.py:209-217` 的 `combine_evaluation`：只有 `threshold != None` 的 component 进 `failed_buf`，而 `mlp_bm_l2c2.py:319-334` 只给 anchor_height_error 传了 threshold）→ w[success] *= 0.999^200, w[fail] = 1.0；不做归一化；失败 metric 直接复用 reward/termination kernel 算，不走 env extras；evaluator 结束后 `env.reset_all()` 恢复训练（v1 不实现 save/restore_state）；落盘 failed_motions_epoch_*.txt
    status: pending
  - id: general-tracking-runner
    content: GeneralTrackingOnPolicyRunner(MjlabOnPolicyRunner)：处理 3-obs-group 收集、update 时传 clean obs 给 PPO.update、evaluator hook（每 200 epoch 调 run_eval + 更新 library.clip_weights）、save_predicted_motion_lib_every=3 checkpoint 落盘（对齐 ProtoMotions）
    status: pending
  - id: g1-env-cfg
    content: G1GeneralTrackingTrainEnvCfg / PlayEnvCfg（tasks/general_tracking/gt_env_cfg.py + config/g1/env_cfgs.py）：anchor_body_name='torso_link'、body_names 从 schema.py 引入、motion_library_path、episode_length_s=20.0 (max_episode_length=1000 @ 50Hz)、num_envs=4096、init_start_prob=0.2
    status: pending
  - id: g1-rl-cfg
    content: G1GeneralTrackingPPOCfg：obs_groups={'actor':('actor_noisy',), 'critic':('critic',), 'actor_clean':('actor_clean',)}，装载 GeneralTrackingActorCriticCfg + GeneralTrackingPPOCfg(l2c2_coef=1.0) + 嵌套 MotionSuccessEvaluatorCfg(eval_metrics_every=200, max_eval_steps=600, success_discount=0.999, failure_discount=0, evaluation_components 中仅 anchor_height_error 带 threshold=0.25，其余为 log-only)
    status: pending
  - id: task-registration
    content: register_mjlab_task('GeneralTracking-Flat-Unitree-G1')；uv run list-envs 可见；bash scripts/train.sh NUM_ENVS=64 MAX_ITER=2 冒烟
    status: pending
  - id: smoke-training
    content: num_envs=256, iter=10 冒烟：无 NaN、所有 obs/reward/termination 计算通过
    status: pending
  - id: small-training
    content: num_envs=1024, iter=500：reward 集合（global_anchor_ori、relative_body_pos 等）稳定下降，L2C2 loss 非零且可观，evaluator 首次触发（若 iter>=200）clip_weights 更新
    status: pending
  - id: full-training
    content: num_envs=4096, iter>=10k：LAFAN1 全 clip 收敛，anchor_height_error < 0.25 的成功率达到 70%+，motion_weights 熵显著下降，失败 clip 维持 w=1 导致被高频采样
    status: pending
  - id: play-eval
    content: bash scripts/play.sh CKPT=<path> 加载；ghost 绘制参考 motion；跑全 LAFAN1 得 per-clip success 报告
    status: pending
  - id: tests-lint
    content: pytest 覆盖 MotionLibrary（采样/未来帧 clamp/manifest-control-fps 校验）、MultiClipMotionCommand、region-density weights（单机器人比对 ProtoMotions 数值）、reward kernel（region 加权正确性）、L2C2（global ratio 公式、coef=0 等价 baseline）、evaluator（单 clip 扫描+权重更新，验证只有带 threshold 的 component 触发 failure，其余 metrics 仅记录）；ruff + pyright 通过
    status: pending
  - id: docs
    content: README、AGENTS.md（继承下游规则）、docs/plans/2026-04-17-general-tracking-design.md（含架构图、数据契约、对齐表、验收条目）
    status: pending
  - id: multi-gpu-v2
    content: v2：chunk_{rank}.npz 目录 + MotionLibrary rank-aware 加载、DDP PPO、evaluator cross-rank reduce；加速比验证
    status: pending
isProject: false
---

# general_tracking — ProtoMotions G1 General Motion Tracker 实现规划 (v3)

## 1. 范围

**做**:

- 下游独立包 `general_tracking`（mjlab 插件风格，非 fork 非 editable install）
- **语义忠实**复现 `ProtoMotions/examples/experiments/mimic/mlp_bm_l2c2.py` 的 G1 general tracker 训练 recipe
- 消费 `/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz/*.npz`（已在 50 Hz，字段与 `mjlab/src/mjlab/tasks/tracking/mdp/commands.py:36-60` 的 MotionLoader 格式一致）
- 完整 reward 集合（7 项）+ noisy/clean reduced-coords actor obs + max-coords critic obs + future_steps=[1,2,4,8] + L2C2(λ=1.0) + MimicEvaluator 风格周期 reweighting
- 扩展 `rsl-rl` PPO（`GeneralTrackingPPO`/`GeneralTrackingOnPolicyRunner`）
- 三层入口：用户 shell launcher (`scripts/*.sh`) → `[project.scripts]` 注册的 Python CLI → Python 模块
- v1 单 GPU；v2 多 GPU chunk 分片

**不做**:

- 不支持 IsaacLab/IsaacGym/Newton/Genesis
- 不做 retargeting（数据已有）
- 不做 ONNX 导出、不做真机部署
- 不走 ProtoMotions 的 `.pt` 格式路径
- v1 不做 contact reward（`ref_contact_smooth_window` 暂不启用）

## 2. 关键差异对齐（核心，务必一次看清）

经源码级核查，ProtoMotions `mlp_bm_l2c2.py` 与 mjlab 自带 `tasks/tracking/` 在以下维度结构性差异。mjlab 可复用：**简单 reward 数学内核、MotionLoader NPZ schema、runner/register 基础设施、部分 DR events**。其余需自写。


| 维度                | mjlab tracking (BM reimpl)                                                                                       | ProtoMotions G1 tracker (`mlp_bm_l2c2.py`)                                                                                                                                                                                                                 | 本项目做法                                                                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Robot/PD          | G1 默认                                                                                                            | G1 默认，`anchor_body_name=torso_link`                                                                                                                                                                                                                        | 复用 `get_g1_robot_cfg()`，anchor 固定 `torso_link`（`mlp_bm_l2c2.py:348`）                                                                             |
| Reward 项数         | 6-7（含 anchor_pos + self_collision）                                                                               | **7**（含 `global_anchor_ori`，4 个 body 项全部 region-weighted，去掉 anchor_pos/self_collision）                                                                                                                                                                     | 7 项，见 §7                                                                                                                                         |
| region weights    | 无                                                                                                                | **kinematic chain density 自动计算**（`compute_body_density_weights`, discount=0.9, 归一化 sum=num_bodies）（`pose_lib.py:204-262`）                                                                                                                                  | 复刻 `compute_body_density_weights` 到 `schema.py`                                                                                                  |
| Action            | `JointPositionAction(scale=0.5, offset=default)`                                                                 | `bm_pd_action`：per-joint scale=`effort_limit/stiffness`（实际实现，非 docstring 的 0.25，见 `action_functions.py:403`）                                                                                                                                               | 子类化 mjlab `JointPositionAction`；scale dict 由 `build_g1_bm_action_scale()` 从 `G1_ARTICULATION.actuators` 推 `e/s`；额外 host 1-slot `_history` buffer |
| Termination       | 4 项（timeout + anchor_pos_z + anchor_ori + body_pos_z）                                                            | 仅 1 项 fall：`anchor_height_error_term_factory(threshold=0.25)`（`mlp_bm_l2c2.py:154`）                                                                                                                                                                        | 单 fall，复用 mjlab kernel                                                                                                                           |
| Actor obs (noisy) | `command(dof) + anchor_pos_b + anchor_ori_b + base_lin_vel + base_ang_vel + joint_pos + joint_vel + last_action` | `noisy_reduced_coords_obs + noisy_mimic_reduced_coords_target_poses([1,2,4,8] × (6+nd+nd)) + historical_previous_processed_actions(1 步 processed)`                                                                                                         | 完全重写                                                                                                                                             |
| Actor obs (clean) | 无                                                                                                                | 同 noisy 三项但 `use_noisy=False`（供 L2C2 配对）                                                                                                                                                                                                                   | 新写 `actor_clean` group                                                                                                                           |
| Critic obs        | actor + body_pos + body_ori（privileged）                                                                          | `max_coords_obs(local_obs=True,root_height=True) + mimic_max_coords_target_poses(with_velocities=True, with_relative=True) + historical_previous_processed_actions`                                                                                        | 完全重写                                                                                                                                             |
| L2C2              | 无                                                                                                                | **λ=1.0**（`mlp_bm_l2c2.py:313`）；对两 obs_pairs（`noisy↔clean` 两项）配对                                                                                                                                                                                           | PIN `lambda_l2c2=1.0`                                                                                                                            |
| PPO / 训练          | rsl-rl 默认（lr=1e-3, epochs=5, grad_clip=1, adaptive_schedule）                                                     | **actor lr=2e-5 / critic lr=1e-4, Adam betas=(0.95,0.99), num_mini_epochs=2, gradient_clip_val=50, adaptive_lr=False, normalize_rewards=False, clip_critic_loss=True, advantage_normalization(shift_mean=True), entropy_coef=0**（`mlp_bm_l2c2.py:297-337`） | 子类化 rsl-rl，重写 update                                                                                                                             |
| Network           | actor/critic `[512,256,128]` elu                                                                                 | **actor 6×1024 ReLU / critic 4×1024 ReLU**，分离参数；`normalize_obs=True, norm_clamp_value=5`；`actor_logstd=-2.9, learnable_std=True`（`mlp_bm_l2c2.py:241-277`）                                                                                                 | 参 ProtoMotions                                                                                                                                   |
| Motion data       | 单 NPZ `MotionLoader`（整数 step）                                                                                    | `MotionLib`（多 `.motion`→`.pt` chunks，**连续时间+插值**）                                                                                                                                                                                                          | **新写 MotionLibrary**：多 NPZ 扁平拼接，Option A 整数 step；manifest `control_fps` vs env control fps 强校验（NPZ 不塞 fps）                                       |
| Motion 采样         | bin-based adaptive（clip 内 sub-range）                                                                             | 初始 `motion_lib.motion_weights` uniform + **MimicEvaluator 周期扫描驱动 reweight**                                                                                                                                                                                | **MotionSuccessEvaluator** 周期扫描                                                                                                                  |
| 时间采样              | 无 init_start_prob                                                                                                | `init_start_prob=0.2`（20% 概率 t=0，80% 均匀）（`mlp_bm_l2c2.py:207`）                                                                                                                                                                                             | 加入                                                                                                                                               |
| Episode length    | 可配                                                                                                               | `max_episode_length=1000 steps` @ 50Hz = **20 秒**（`mlp_bm_l2c2.py:199`）                                                                                                                                                                                    | 20s                                                                                                                                              |
| DR                | friction+COM+push+encoder_bias                                                                                   | 上述 + **action noise + reset noise**                                                                                                                                                                                                                        | 扩展                                                                                                                                               |


**权威数值定值（不留"待定"）**：

- `lambda_l2c2 = 1.0`（`mlp_bm_l2c2.py:313`）
- `actor_logstd = -2.9`（init std ≈ 0.055，学习型），`learnable_std=True`（`:243-244`）
- `gradient_clip_val = 50.0`, `num_mini_epochs = 2`, `entropy_coef = 0`, `clip_critic_loss = True`（`:307, 309, 310`）
- `actor_optim = Adam(lr=2e-5, betas=(0.95,0.99))`, `critic_optim = Adam(lr=1e-4, betas=(0.95,0.99))`（`:297-302`）
- `eval_metrics_every = 200`（默认，`evaluators/config.py:37-40`）
- `max_eval_steps = 600`（12 s @ 50Hz）
- `motion_weights_update_success_discount = 0.999`, `motion_weights_update_failure_discount = 0`（`mlp_bm_l2c2.py:331-332`）
- `init_start_prob = 0.2`（`:207`）
- `max_episode_length = 1000`（20 s，`:199`）
- `future_steps = [1, 2, 4, 8]`（`:105`）
- `anchor_height_error threshold = 0.25`（`:154`）
- `advantage_normalization: enabled=True, shift_mean=True`（`:335-337`）
- `normalize_obs=True, norm_clamp_value=5`（`:257-258, 273-274`）
- `history_steps=1, processed=True`（上一步 tanh/clamp 后动作，非多步 history；`:147-149, 200`）
- `region_weights` 来自 `compute_body_density_weights`（kinematic tree, discount=0.9；`pose_lib.py:204-262`）

## 3. 架构图

```mermaid
flowchart TB
    subgraph entry [User entry points]
        shellTrain["scripts/train.sh"]
        shellPlay["scripts/play.sh"]
        shellBuild["scripts/build_manifest.sh"]
        shellTrain --> gtTrain["uv run gt-train"]
        shellPlay --> gtPlay["uv run gt-play"]
        shellBuild --> gtBuild["uv run gt-build-manifest"]
    end

    subgraph data [Data Layer]
        npzDir["lafan1_npz/*.npz (fps=50)"]
        manifest["motion_manifest.yaml<br/>(path/weight/fps/num_frames)"]
        schema["robots/g1/schema.py<br/>anchor=torso_link<br/>body_names (frozen)<br/>density_weights"]
        gtBuild --> manifest
        npzDir --> manifest
        manifest -->|fps assert| motionLib["MotionLibrary<br/>flat tensors<br/>clip_starts/lengths/weights<br/>init_start_prob=0.2<br/>INTEGER time_steps"]
        schema --> motionLib
    end

    subgraph env [Env Layer: general_tracking task]
        motionLib --> cmd["MultiClipMotionCommand<br/>clip_ids + time_steps per env<br/>future_offsets=[1,2,4,8]"]
        cmd --> actorNoisy["actor_noisy group:<br/>reduced_coords_obs (noisy)<br/>reduced_coords_target_poses (noisy)<br/>processed_action_history (1 step)"]
        cmd --> actorClean["actor_clean group:<br/>same 3 terms, enable_corruption=False"]
        cmd --> criticG["critic group:<br/>max_coords_obs<br/>max_coords_target_poses<br/>processed_action_history"]
        cmd --> rewards["7 reward terms:<br/>global_anchor_ori(0.5,σ0.4)<br/>rel_body_pos(1.0,σ0.3,RW)<br/>rel_body_ori(1.0,σ0.4,RW)<br/>body_lin_vel(1.0,σ1.0,RW)<br/>body_ang_vel(1.0,σ3.14,RW)<br/>action_rate(-0.1)<br/>joint_pos_limits(-10)"]
        cmd --> term["Termination: fall<br/>anchor_height_err>0.25"]
    end

    subgraph learn [Learning Layer: learning/ppo]
        actorNoisy --> policy["Actor: 6×1024 ReLU<br/>init_logstd=-2.9, learnable"]
        actorClean --> policy
        criticG --> value["Critic: 4×1024 ReLU"]
        policy --> l2c2["L2C2 loss (λ=1.0)<br/>scalar output²/input².detach()<br/>agent.py:497-524"]
        policy --> ppo["GeneralTrackingPPO update:<br/>actor Adam lr=2e-5<br/>critic Adam lr=1e-4<br/>betas=(0.95,0.99)<br/>epochs=2, grad_clip=50<br/>entropy=0, adaptive_lr=off"]
        value --> ppo
        l2c2 --> ppo
    end

    subgraph evalSweep [Evaluator Layer - periodic sweep]
        ppo -->|every 200 epochs| evaluator["MotionSuccessEvaluator.run_eval()<br/>pause training<br/>scan ALL clips systematically<br/>per-clip boolean pass/fail"]
        evaluator -->|w[succ] *= 0.999^200| motionLib
        evaluator -->|w[fail] = 1.0| motionLib
        evaluator -.->|dump| failedLog["failed_motions_epoch_*.txt"]
    end

    gtTrain --> runner["GeneralTrackingOnPolicyRunner<br/>(mjlab train CLI compatible)"]
    ppo --> runner
    evaluator --> runner
    gtPlay -.-> runner
```



## 4. 项目布局

```
controller/general_tracking/                     # 仓库根
├── pyproject.toml                               # mjlab Git pin; [project.scripts] gt-train/gt-play/gt-build-manifest
├── README.md
├── AGENTS.md
├── LICENSE
├── Makefile
├── scripts/                                     # 用户入口（shell launchers, thin wrappers）
│   ├── train.sh                                 # uv run gt-train ...
│   ├── play.sh                                  # uv run gt-play ...
│   └── build_manifest.sh                        # uv run gt-build-manifest ...
├── src/general_tracking/
│   ├── __init__.py                              # imports .tasks.* to trigger registration
│   ├── robots/
│   │   └── g1/
│   │       ├── __init__.py
│   │       └── schema.py                        # body_names, anchor_idx, density_weights (FROZEN)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── motion_library.py                    # MotionLibrary (integer step, fps assert)
│   │   ├── manifest.py                          # load/save motion_manifest.yaml
│   │   └── cli/
│   │       ├── __init__.py
│   │       └── build_manifest.py                # def main() → registered as gt-build-manifest
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── general_tracking/                    # task family dir
│   │       ├── __init__.py
│   │       ├── gt_env_cfg.py                    # make_general_tracking_env_cfg()
│   │       ├── mdp/
│   │       │   ├── __init__.py
│   │       │   ├── commands.py                  # MultiClipMotionCommand(Cfg)
│   │       │   ├── actions.py                   # BMPositionAction(Cfg)
│   │       │   ├── observations.py              # reduced_*, max_*, processed_action_history
│   │       │   ├── rewards.py                   # region_weighted_* kernels
│   │       │   ├── terminations.py              # alias for mjlab anchor_height_error
│   │       │   └── events.py                    # action_noise, reset_noise
│   │       └── config/
│   │           └── g1/
│   │               ├── __init__.py              # register_mjlab_task(...)
│   │               ├── env_cfgs.py              # Train/Play env cfgs (G1-specific overrides)
│   │               └── rl_cfg.py                # G1GeneralTrackingPPOCfg
│   ├── learning/
│   │   ├── __init__.py
│   │   └── ppo/
│   │       ├── __init__.py
│   │       ├── l2c2.py                          # compute_l2c2_loss()
│   │       ├── actor_critic.py                  # GeneralTrackingActorCritic
│   │       ├── ppo.py                           # GeneralTrackingPPO (子类化 rsl-rl)
│   │       ├── evaluator.py                     # MotionSuccessEvaluator (periodic sweep)
│   │       └── runner.py                        # GeneralTrackingOnPolicyRunner
│   └── cli/
│       ├── __init__.py
│       ├── train.py                             # def main() → gt-train
│       └── play.py                              # def main() → gt-play
├── tests/
│   ├── test_motion_library.py
│   ├── test_multi_clip_command.py
│   ├── test_reduced_coords_obs.py
│   ├── test_reduced_coords_target_poses.py
│   ├── test_max_coords_obs.py
│   ├── test_processed_action_history.py
│   ├── test_region_density_weights.py           # 数值对比 ProtoMotions
│   ├── test_region_weighted_rewards.py
│   ├── test_l2c2_loss.py                        # λ=0 等价 baseline
│   ├── test_motion_success_evaluator.py         # 单 clip 扫描 + 权重更新
│   └── test_task_registration.py
└── docs/
    └── plans/
        └── 2026-04-17-general-tracking-design.md
```

### 4.1 三层入口契约

```
user types: bash scripts/train.sh [NUM_ENVS=...] [MAX_ITER=...] [MOTION_LIB_PATH=...]
          ↓
scripts/train.sh (thin bash wrapper):
          uv run gt-train --task GeneralTracking-Flat-Unitree-G1 \
                          --num-envs $NUM_ENVS --max-iterations $MAX_ITER ...
          ↓
pyproject.toml [project.scripts] maps:
          gt-train = "general_tracking.cli.train:main"
          ↓
src/general_tracking/cli/train.py::main()
  → 复用 mjlab 训练入口（可直接 delegate 到 mjlab.train:main 或包装 argparse 后再调）
  → 实例化 GeneralTrackingOnPolicyRunner
```

`pyproject.toml` 关键片段：

```toml
[project]
name = "general_tracking"
dependencies = ["mjlab[cu128]>=1.3,<1.4"]

[project.scripts]
gt-train          = "general_tracking.cli.train:main"
gt-play           = "general_tracking.cli.play:main"
gt-build-manifest = "general_tracking.data.cli.build_manifest:main"

[project.entry-points."mjlab.tasks"]
general_tracking = "general_tracking.tasks.general_tracking.config.g1"

# 只有确需未发布 upstream 特性时才加这一段
[tool.uv.sources]
mjlab = { git = "https://github.com/mujocolab/mjlab", rev = "<commit>" }
```

Shell launcher 示例（`scripts/train.sh`）：

```bash
#!/usr/bin/env bash
set -euo pipefail
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-30000}"
MOTION_LIB_PATH="${MOTION_LIB_PATH:-$HOME/Downloads/Data/G1_retargeted/lafan1_npz/motion_manifest.yaml}"
export MOTION_LIB_PATH
exec uv run gt-train \
  --task GeneralTracking-Flat-Unitree-G1 \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITER" \
  "$@"
```

## 5. 数据契约（Phase 1 冻结）

### 5.1 Body / anchor schema（`robots/g1/schema.py`）

```python
ANCHOR_BODY_NAME: str = "torso_link"   # ProtoMotions G1 default, mlp_bm_l2c2.py:348

BODY_NAMES: tuple[str, ...] = (
    # 按 mjlab G1 MJCF body 实际顺序冻结；Phase 1 的 freeze-body-schema
    # 任务从 MJCF 提取一次写死，之后 NPZ body 维顺序必须与此一致。
    # 占位，Phase 1 精确落盘。
)

NUM_BODIES: int = len(BODY_NAMES)
ANCHOR_BODY_INDEX: int = BODY_NAMES.index(ANCHOR_BODY_NAME)

JOINT_NAMES: tuple[str, ...] = (...)   # 同上冻结
NUM_DOFS: int = len(JOINT_NAMES)

CONTROL_FPS: float = 50.0   # mjlab env.step_dt = 0.02s 对应 50 Hz

def compute_body_density_weights(
    parent_indices: torch.Tensor,
    local_pos: torch.Tensor,
    discount: float = 0.9,
) -> torch.Tensor:
    """复刻 ProtoMotions pose_lib.py:204-262。
    返回 [num_bodies]，sum 归一到 num_bodies。"""
    ...

DENSITY_WEIGHTS: torch.Tensor = compute_body_density_weights(
    parent_indices=..., local_pos=..., discount=0.9,
)   # 懒加载/cache；供 region_weighted reward kernel 引用
```

**Phase 1 第一个任务 `freeze-body-schema`** 必须：

1. 从 mjlab G1 MJCF 枚举 body/joint 列表并写入 `schema.py`；
2. 用 `lafan1_npz` 里任意一个 NPZ 验证 `body_pos_w.shape[1] == NUM_BODIES`、`joint_pos.shape[1] == NUM_DOFS`；不匹配即硬错（证明数据与 MJCF 不配套）；
3. 通过 `compute_body_density_weights` 离线算一次 `DENSITY_WEIGHTS`，与 ProtoMotions 同函数在等价 kinematic tree 上的输出做数值 diff（要求 < 1e-5）。

任何后续任务（observation、reward、command）都必须 `from general_tracking.robots.g1.schema import ANCHOR_BODY_NAME, BODY_NAMES, DENSITY_WEIGHTS`，不自己硬编码。

### 5.2 NPZ 格式（已有，不变）

字段（与 [mjlab/src/mjlab/tasks/tracking/mdp/commands.py:40-59](mjlab/src/mjlab/tasks/tracking/mdp/commands.py) 一致）：

- `joint_pos: (T, NUM_DOFS)`
- `joint_vel: (T, NUM_DOFS)`
- `body_pos_w: (T, NUM_BODIES, 3)`
- `body_quat_w: (T, NUM_BODIES, 4)` wxyz
- `body_lin_vel_w: (T, NUM_BODIES, 3)`
- `body_ang_vel_w: (T, NUM_BODIES, 3)`

这里 **不向 NPZ 内再塞 `fps` 字段**。`control_fps` 作为数据集级契约写在 manifest 中，由 `MotionLibrary` 在加载时与 env control dt 做一致性校验。

### 5.3 Manifest 格式

```yaml
version: 1
control_fps: 50.0
clips:
  - path: walk_01.npz
    weight: 1.0
    num_frames: 523
  - path: dance_funky.npz
    weight: 1.2
    num_frames: 1024
```

`gt-build-manifest`（位于 `general_tracking/data/cli/build_manifest.py`）扫目录，逐 NPZ `np.load` 读取帧数并写入相对路径。`control_fps` 由 CLI 参数注入，例如：

```bash
uv run gt-build-manifest --input-dir ~/Downloads/Data/G1_retargeted/lafan1_npz --control-fps 50.0
```

`MotionLibrary` 初始化时必须执行：

```python
assert abs(manifest.control_fps - env_control_fps) < 1e-6, (
    f"manifest.control_fps={manifest.control_fps} != env_control_fps={env_control_fps}"
)
```

### 5.4 MotionLibrary API（Option A：整数 step）

```python
class MotionLibrary:
    # 设备常驻扁平 tensor（按 schema 顺序）
    joint_pos: Tensor        # (total_frames, NUM_DOFS)
    joint_vel: Tensor
    body_pos_w: Tensor       # (total_frames, NUM_BODIES, 3)
    body_quat_w: Tensor      # (total_frames, NUM_BODIES, 4) wxyz
    body_lin_vel_w: Tensor
    body_ang_vel_w: Tensor

    # 每 clip 元数据
    clip_starts: Tensor      # (n_clips,)
    clip_lengths: Tensor     # (n_clips,)
    clip_weights: Tensor     # (n_clips,) 动态更新；初始化从 manifest.weight
    control_fps: float = 50.0

    @property
    def num_clips(self) -> int: ...
    @property
    def num_frames(self) -> int: ...

    def sample_clip_ids(self, n: int) -> Tensor:
        return torch.multinomial(self.clip_weights, n, replacement=True)

    def sample_init_time(self, clip_ids: Tensor, init_start_prob: float = 0.2) -> Tensor:
        """对齐 ProtoMotions MimicMotionManager.sample_motions (motion_manager.py:409-417)"""
        lengths = self.clip_lengths[clip_ids]
        max_time = (lengths - 1).clamp_min(1)
        uniform_t = (torch.rand_like(max_time, dtype=torch.float32) * max_time).long()
        start_mask = torch.rand_like(max_time, dtype=torch.float32) < init_start_prob
        return torch.where(start_mask, torch.zeros_like(uniform_t), uniform_t)

    def get_state_at(self, clip_ids: Tensor, time_steps: Tensor) -> dict[str, Tensor]:
        """time_steps 为整数 step；直接整数索引，NO interpolation（Option A 不变量）"""
        flat = self.clip_starts[clip_ids] + time_steps.clamp(min=0, max=self.clip_lengths[clip_ids]-1)
        return {"joint_pos": self.joint_pos[flat], ...}

    def get_future_states(self, clip_ids: Tensor, time_steps: Tensor, offsets: Sequence[int]) -> dict[str, Tensor]:
        """返回 (n_env, n_offsets, ...) 每个 field；末尾 clamp 到 clip_lengths-1"""
        ...

    def update_weights(self, new_weights: Tensor) -> None:
        """by MotionSuccessEvaluator. 不做 sum 归一化（对齐 ProtoMotions）"""
        assert new_weights.shape == self.clip_weights.shape
        self.clip_weights[:] = new_weights
```

关键不变量（在 `__init__` 里 assert，不只是文档）：

```python
assert abs(manifest.control_fps - env_control_fps) < 1e-6, (
    f"manifest.control_fps={manifest.control_fps} != env_control_fps={env_control_fps}"
)
```

## 6. Observations（完全重写）

### 6.1 Actor noisy group（对齐 ProtoMotions `mlp_bm_l2c2.py:112-133`）

**Term 1: `reduced_coords_obs(env)`**

对齐 `reduced_coords_obs_factory(use_noisy=True, root_height_obs=False, root_vel_obs=False)`。输出：`[dof_pos, dof_vel, root_local_ang_vel, proj_gravity_from_anchor]`。

```python
def reduced_coords_obs(env) -> Tensor:
    robot = env.scene["robot"]
    anchor_quat_w = robot.data.body_quat_w[:, ANCHOR_BODY_INDEX]
    return torch.cat([
        robot.data.joint_pos,                               # nd
        robot.data.joint_vel,                               # nd
        robot.data.root_ang_vel_b,                          # 3
        projected_gravity_from_quat(anchor_quat_w),         # 3
    ], dim=-1)
```

Noise（参考 ProtoMotions `EnvContext.noisy.*` 经验值）：

- `joint_pos`: `Unoise(-0.02, 0.02)`
- `joint_vel`: `Unoise(-0.1, 0.1)`
- `root_ang_vel_b`: `Unoise(-0.02, 0.02)`
- `proj_gravity`: `Unoise(-0.05, 0.05)`

**Term 2: `reduced_coords_target_poses(env, command_name, offsets=(1,2,4,8))`**

对齐 `mimic_target_poses_reduced_coords_factory(use_noisy=True, include_dof_vel=True, include_xy_offset=False)`。底层 `build_reduced_coords_target_poses`（`target_poses.py:354-456`）每个 future 步输出顺序为：

1. `target_anchor_rot_obs` = `quat_to_tan_norm(rel_quat)`（6D）
2. `ref_state_dof_vel`（nd）
3. `ref_state_dof_pos`（nd）

```python
def reduced_coords_target_poses(env, command_name: str, offsets=(1,2,4,8)) -> Tensor:
    cmd = env.command_manager.get_term(command_name)
    futures = cmd.library.get_future_states(cmd.clip_ids, cmd.time_steps, offsets)
    robot = env.scene["robot"]
    cur_anchor_quat = robot.data.body_quat_w[:, ANCHOR_BODY_INDEX]

    pieces = []
    for k, off in enumerate(offsets):
        ref_anchor_quat = futures["body_quat_w"][:, k, ANCHOR_BODY_INDEX]
        rel_quat = quat_mul(quat_conjugate(cur_anchor_quat), ref_anchor_quat)
        rel_rot_6d = quat_to_tan_norm(rel_quat)
        pieces.extend([
            rel_rot_6d,
            futures["joint_vel"][:, k],
            futures["joint_pos"][:, k],
        ])
    return torch.cat(pieces, dim=-1)   # (n_env, len(offsets) * (6 + 2*nd))
```

Noise：`rel_rot_6d` `Unoise(-0.02,0.02)`，`target_dof_pos` `Unoise(-0.02,0.02)`，`target_dof_vel` `Unoise(-0.1,0.1)`。

**Term 3: `processed_action_history(env, history_steps=1)`**

对齐 `previous_actions_factory(history_steps=1, processed=True)`。**只有 1 步，不是多步 history**。

```python
def processed_action_history(env) -> Tensor:
    """读 BMPositionAction 内部维护的 1-slot buffer。"""
    return env.action_manager.get_term("joint_pos")._history   # (n_env, nd)
```

**Owner 明确**：由 `BMPositionAction`（§9，subclass `JointPositionAction`）持有 `_history`；父类 `process_actions()` 跑完写完 `self._processed_actions` 后，子类再 `self._history.copy_(self._processed_actions)`；`reset(env_ids)` 清零对应槽位。不依赖 mjlab `action_manager.prev_processed_actions`——该字段当前版本并不暴露 "processed" 历史。

无 noise（`actor_clean` / `critic` group 中同样读取该 buffer，只是上层 group `enable_corruption=False`）。

### 6.2 Actor clean group（L2C2 配对）

与 actor_noisy 完全同源三个 term，但 `ObservationGroupCfg(enable_corruption=False)` 关闭所有 Unoise。

### 6.3 Critic group（privileged）

**Term 1: `max_coords_obs(env)`**

对齐 `max_coords_obs_factory(local_obs=True, root_height_obs=True, observe_contacts=False)`。

```python
def max_coords_obs(env) -> Tensor:
    robot = env.scene["robot"]
    body_pos_w  = robot.data.body_link_pos_w
    body_quat_w = robot.data.body_link_quat_w
    body_vel_w  = robot.data.body_link_lin_vel_w
    body_av_w   = robot.data.body_link_ang_vel_w
    root_pos  = body_pos_w[:, 0]
    root_quat = body_quat_w[:, 0]

    heading_inv = calc_heading_quat_inv(root_quat)
    local_pos  = quat_rotate_batch(heading_inv, body_pos_w - root_pos[:, None])
    local_quat = quat_mul_batch(heading_inv, body_quat_w)
    local_vel  = quat_rotate_batch(heading_inv, body_vel_w)
    local_av   = quat_rotate_batch(heading_inv, body_av_w)
    root_h = root_pos[:, 2:3]
    return torch.cat([root_h, local_pos.flatten(1), local_quat.flatten(1),
                      local_vel.flatten(1), local_av.flatten(1)], dim=-1)
```

**Term 2: `max_coords_target_poses(env, command_name, offsets=(1,2,4,8))`**

对齐 `mimic_target_poses_max_coords_factory(with_velocities=True, with_relative=True)`。每 future 步输出 `[abs_pos, abs_quat_tan_norm, rel_pos, rel_quat_tan_norm, abs_vel, abs_ang_vel]`（全部局部化）。具体 layout 以 ProtoMotions `target_poses.py:158-340` 为准。

**Term 3: `processed_action_history`（同 actor）**

critic group `enable_corruption=False`。

## 7. Reward 重写（7 项）

### 7.1 Region-density weights（kernel 依赖）

```python
# tasks/general_tracking/mdp/rewards.py
from general_tracking.robots.g1.schema import DENSITY_WEIGHTS

def _apply_region_weights(per_body_rew: Tensor) -> Tensor:
    """per_body_rew: (n_env, NUM_BODIES). Returns (n_env,) weighted mean."""
    return (per_body_rew * DENSITY_WEIGHTS).sum(-1) / DENSITY_WEIGHTS.sum()
```

### 7.2 Reward 集合（对齐 `mlp_bm_l2c2.py:158-195`）

```python
rewards = {
    "motion_global_anchor_ori": RewardTermCfg(
        func=mdp.motion_global_anchor_orientation_error_exp,   # 复用 mjlab kernel
        weight=0.5, params={"command_name": "motion", "std": 0.4},
    ),
    "motion_relative_body_pos": RewardTermCfg(
        func=region_weighted_relative_body_position_error_exp,
        weight=1.0, params={"command_name": "motion", "std": 0.3},
    ),
    "motion_relative_body_ori": RewardTermCfg(
        func=region_weighted_relative_body_orientation_error_exp,
        weight=1.0, params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
        func=region_weighted_global_body_linear_velocity_error_exp,
        weight=1.0, params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
        func=region_weighted_global_body_angular_velocity_error_exp,
        weight=1.0, params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_pos_limits": RewardTermCfg(
        func=mdp.joint_pos_limits, weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    ),
}
```

**明确不加**：`motion_global_anchor_position_error_exp`、`self_collision_cost`、任何 contact-based reward。

### 7.3 Kernel 实现要点

```python
def region_weighted_relative_body_position_error_exp(env, command_name: str, std: float) -> Tensor:
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    anchor_pos = robot.data.body_link_pos_w[:, ANCHOR_BODY_INDEX]
    cur_rel = robot.data.body_link_pos_w - anchor_pos[:, None]
    tgt_anchor_pos = cmd.body_pos_w[:, ANCHOR_BODY_INDEX]
    tgt_rel = cmd.body_pos_w - tgt_anchor_pos[:, None]
    sq_err = (cur_rel - tgt_rel).pow(2).sum(-1)   # (n_env, NUM_BODIES)
    per_body_rew = torch.exp(-sq_err / std**2)
    return _apply_region_weights(per_body_rew)
```

其余三项 kernel 按 `relative_body_ori`（四元数距离）、`global_body_lin_vel`、`global_body_ang_vel` 类似写法构造。

## 8. Termination（单 fall）

```python
terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fall": TerminationTermCfg(
        func=mdp.motion_anchor_height_error_termination,
        params={"command_name": "motion", "threshold": 0.25},
    ),
}
```

## 9. Action

**决策**：子类化 mjlab `[JointPositionAction](mjlab/src/mjlab/envs/mdp/actions/actions.py)` 复用 `process_actions` 数学（`_processed_actions = _raw_actions * _scale + _offset`，`actions.py:154-162`）；只在子类中额外 host 1-slot `_history` buffer（供 `processed_action_history` observation 读取）。**不再重写** process_actions 内部算式——对齐上轮 `reuse_mjlab` 决策。

per-joint scale 由 `build_g1_bm_action_scale()` 从 mjlab `G1_ARTICULATION.actuators` 解析出 `effort_limit / stiffness`，**不乘 0.25**（对齐 ProtoMotions `action_functions.py:403` 实际代码，不是 mjlab `G1_ACTION_SCALE` 的 `0.25 * e / s`）。offset 走父类的 `use_default_offset=True` 路径，等价 `entity.data.default_joint_pos[target_ids].clone()`。

```python
# src/general_tracking/robots/g1/action_scale.py
from mjlab.asset_zoo.robots.unitree_g1 import G1_ARTICULATION
from mjlab.actuator import BuiltinPositionActuatorCfg

def build_g1_bm_action_scale() -> dict[str, float]:
    """为 JointPositionActionCfg.scale 构造 joint_regex → scale 字典。

    scale = effort_limit / stiffness per actuator group (ProtoMotions bm_pd_action)。
    严禁直接使用 mjlab 的 G1_ACTION_SCALE（它是 0.25*e/s，对齐原版 BeyondMimic 而非 ProtoMotions）。
    """
    result: dict[str, float] = {}
    for a in G1_ARTICULATION.actuators:
        assert isinstance(a, BuiltinPositionActuatorCfg)
        assert a.effort_limit is not None and a.stiffness is not None
        for joint_regex in a.target_names_expr:
            result[joint_regex] = a.effort_limit / a.stiffness
    return result
```

```python
# src/general_tracking/tasks/general_tracking/mdp/actions.py
from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

@dataclass(kw_only=True)
class BMPositionActionCfg(JointPositionActionCfg):
    """与父类唯一差异：class_type=BMPositionAction；scale dict 由 cfg 显式传入（e/s，非 0.25*e/s）。
    use_default_offset=True → offset = entity.data.default_joint_pos[target_ids].clone() (parent)。
    无 tanh，无 clamp。"""

    def build(self, env):
        return BMPositionAction(self, env)


class BMPositionAction(JointPositionAction):
    """复用父类的 process_actions 数学 (`_raw * _scale + _offset`)。
    额外维护 1-slot `_history` 供 `processed_action_history` observation 读取。"""

    def __init__(self, cfg: BMPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._history = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._history.copy_(self._processed_actions)   # expose as "previous processed action"

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids] = 0.0
```

Env cfg 注入（`config/g1/env_cfgs.py`）：

```python
cfg.actions["joint_pos"] = BMPositionActionCfg(
    asset_name="robot",
    joint_names=(".*",),
    scale=build_g1_bm_action_scale(),
    use_default_offset=True,
    clip=None,
)
```

`processed_action_history` observation 函数（§6.1 Term 3）直接读 `env.action_manager.get_term("joint_pos")._history`。

## 10. DR + Events

- `geom_friction`（复用 mjlab）
- `body_com_offset`（复用）
- `push_by_setting_velocity`（复用）
- `encoder_bias`（复用）
- **action_noise_event**（新增）：每 step 给 `processed_actions` 加 `Unoise(-0.05, 0.05)`
- **reset_noise_event**（新增）：reset 时给 qpos `Unoise(-0.02, 0.02)`、qvel `Unoise(-0.1, 0.1)`

## 11. PPO 扩展（位于 `learning/ppo/`）

### 11.1 GeneralTrackingActorCritic

```python
@dataclass
class GeneralTrackingActorCriticCfg(RslRlPpoActorCriticCfg):
    actor_hidden_dims: list[int] = field(default_factory=lambda: [1024]*6)
    critic_hidden_dims: list[int] = field(default_factory=lambda: [1024]*4)
    activation: str = "relu"
    init_noise_std: float = math.exp(-2.9)
    learnable_std: bool = True
```

### 11.2 `compute_l2c2_loss`（`learning/ppo/l2c2.py`）

逐字对齐 ProtoMotions `[agents/ppo/agent.py:497-524](ProtoMotions/protomotions/agents/ppo/agent.py)`。三个关键点（不是常数倍差异）：

1. **全量均方标量**：`input_dist` 是把所有 obs_pairs 的元素差平方累加后除以总元素数得到的 scalar（`sum / numel == mean over all elements`）；`output_dist` 同理是 `(mu_n - mu_c).pow(2).mean()` 的 scalar。**不是** per-sample 向量再 `.mean()`。
2. `**input_dist.detach()`**：分母不参与反向。如果不 detach，L2C2 会反过来鼓励 obs pipeline 放大 input 差来稀释惩罚，语义性破坏。
3. **二维加权**：高维 obs pair（`target_poses` 比 `reduced_coords_obs` dim 大）在 `input_dist` 里按元素数自动加权，不是两对等权。

```python
import torch
from torch import Tensor

def compute_l2c2_loss(
    mu_noisy: Tensor, mu_clean: Tensor,
    obs_pairs: list[tuple[Tensor, Tensor]],
    lambda_coef: float,
    eps: float = 1e-8,
) -> Tensor:
    """对齐 ProtoMotions agents/ppo/agent.py:497-524。

    obs_pairs 两项（batch_td[key] ↔ batch_td[clean_key]）：
      (noisy_reduced_coords_obs, clean_reduced_coords_obs)
      (noisy_mimic_reduced_coords_target_poses, clean_mimic_reduced_coords_target_poses)
    """
    input_ss = torch.tensor(0.0, device=mu_noisy.device)
    input_n = 0
    for obs_n, obs_c in obs_pairs:
        diff = obs_n - obs_c
        input_ss = input_ss + diff.pow(2).sum()
        input_n += diff.numel()
    input_dist = (input_ss / max(input_n, 1)).detach()
    output_dist = (mu_noisy - mu_clean).pow(2).mean()
    return lambda_coef * output_dist / (input_dist + eps)
```

单测覆盖：

- lambda=0 时 l2c2 为 0，等价 baseline；
- 与 ProtoMotions 参考实现在同 batch 上逐元素 match；
- 验证 `input_dist.requires_grad is False` 且不在计算图里（可通过 detach 后 retain_graph 反向验证）；
- 二维加权：构造 dim=10 / dim=100 两对，期望 `input_dist = (ss10 + ss100) / 110`，而非 `(mean10 + mean100)`。

### 11.3 GeneralTrackingPPO（`learning/ppo/ppo.py`）

```python
class GeneralTrackingPPO(rsl_rl.algorithms.PPO):
    def __init__(self, actor_critic, *, num_learning_epochs=2, num_mini_batches=4,
                 actor_learning_rate=2e-5, critic_learning_rate=1e-4,
                 l2c2_coef=1.0, gradient_clip_val=50.0,
                 clip_param=0.2, value_loss_coef=1.0, entropy_coef=0.0,
                 gamma=0.99, lam=0.95, schedule="fixed",
                 use_clipped_value_loss=True, **kw):
        super().__init__(actor_critic, learning_rate=actor_learning_rate, ...)
        self.optim_actor  = torch.optim.Adam(self.actor_critic.actor.parameters(),
                                             lr=actor_learning_rate,  betas=(0.95, 0.99))
        self.optim_critic = torch.optim.Adam(self.actor_critic.critic.parameters(),
                                             lr=critic_learning_rate, betas=(0.95, 0.99))
        self.optimizer = None
        self.l2c2_coef = l2c2_coef
        self.gradient_clip_val = gradient_clip_val

    def update(self):
        for ep in range(self.num_learning_epochs):
            for batch in self.storage.mini_batch_generator(self.num_mini_batches, ep):
                obs_actor_noisy = batch.obs_groups["actor"]
                obs_actor_clean = batch.obs_groups["actor_clean"]
                obs_critic      = batch.obs_groups["critic"]

                mu_noisy, logp, entropy = self.actor_critic.act(obs_actor_noisy)
                values = self.actor_critic.evaluate(obs_critic)
                pl, vl, eb = compute_ppo_losses(
                    batch, mu_noisy, logp, values, entropy,
                    clip_param=self.clip_param, use_clipped_value_loss=True,
                    normalize_advantages=True, shift_mean=True,
                )

                mu_clean, _, _ = self.actor_critic.act(obs_actor_clean)
                l2c2 = compute_l2c2_loss(
                    mu_noisy, mu_clean,
                    obs_pairs=[
                        (obs_actor_noisy["reduced"],      obs_actor_clean["reduced"]),
                        (obs_actor_noisy["target_poses"], obs_actor_clean["target_poses"]),
                    ],
                    lambda_coef=self.l2c2_coef,
                )

                total = pl + self.value_loss_coef * vl - self.entropy_coef * eb + l2c2

                self.optim_actor.zero_grad(); self.optim_critic.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(),  self.gradient_clip_val)
                nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.gradient_clip_val)
                self.optim_actor.step();  self.optim_critic.step()
```

## 12. MotionSuccessEvaluator（`learning/ppo/evaluator.py`，periodic sweep）

**关键语义修正**：

1. **不是训练 rollout 的被动监听器**，而是**暂停训练、系统扫描全 clip** 的独立 evaluator；对齐 `mimic_evaluator.py:78-130`、`base_evaluator.py:118-142`。
2. **pass/fail 判据**：逐字对齐 `base_evaluator.py:301-318` + `envs/base_env/utils.py:173-219`（`combine_evaluation`）——**failure 语义完全由 evaluation component 自身是否带 `threshold` 决定**。在 `mlp_bm_l2c2.py:319-334` 默认 recipe 下，6 个注册 component 里只有 `anchor_height_error_metric_factory(threshold=0.25)` 有显式 threshold，其他 5 个工厂 `threshold` 默认 `None`。所以实际 fail/reweight 判据就是 `anchor_height_error > 0.25`，其他 5 个只记录值做训练日志。
3. **metric 计算位置**：evaluator 内部直接调用 reward/termination kernel 的误差函数（`motion_anchor_height_error`、`max_joint_pos_error` 等），**不走** env `extras["eval/*"]`——避免给 env 加只在 eval 时有意义的 observation/extras。
4. **v1 env 状态策略**：扫完后 `env.reset_all()`，放弃当前 rollout；不实现 `save_state/restore_state`（ProtoMotions 的 `env.save_state` 在 mjlab 里不存在）。每 200 epoch 牺牲一个 rollout 可接受；v2 再考虑 snapshot 或独立 eval env。

```python
@dataclass
class MotionSuccessEvaluatorCfg:
    eval_metrics_every: int = 200
    max_eval_steps: int = 600
    success_discount: float = 0.999
    failure_discount: float = 0.0
    # evaluation_components 中只有带 threshold 的项参与 fail/reweight
    evaluation_components: dict[str, EvalMetricSpec] = field(default_factory=lambda: {
        "anchor_ori": EvalMetricSpec(log_only=True),
        "relative_body_pos": EvalMetricSpec(log_only=True),
        "anchor_height_error": EvalMetricSpec(threshold=0.25),
        "gt_error": EvalMetricSpec(log_only=True),
        "gr_error": EvalMetricSpec(log_only=True),
        "max_joint_error": EvalMetricSpec(log_only=True),
    })


class MotionSuccessEvaluator:
    def __init__(self, env, actor_critic, library: MotionLibrary,
                 cfg: MotionSuccessEvaluatorCfg, log_dir: Path):
        self.env = env
        self.actor_critic = actor_critic
        self.library = library
        self.cfg = cfg
        self.log_dir = log_dir
        self.eval_count = 0

    @torch.no_grad()
    def run_eval(self, current_epoch: int) -> dict[str, float]:
        """Pause training, systematically play every clip from t=0, record per-clip fail."""
        self.actor_critic.train(False)

        num_clips = self.library.num_clips
        num_envs = self.env.num_envs
        device = self.env.device
        motion_failed = torch.zeros(num_clips, dtype=torch.bool, device=device)
        log_acc: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.cfg.evaluation_components.keys()
        }

        for start in range(0, num_clips, num_envs):
            end = min(start + num_envs, num_clips)
            batch_clip_ids = torch.arange(start, end, device=device)
            batch_env_ids = torch.arange(len(batch_clip_ids), device=device)

            cmd = self.env.command_manager.get_term("motion")
            cmd.clip_ids[batch_env_ids] = batch_clip_ids
            cmd.time_steps[batch_env_ids] = 0
            obs, _ = self.env.reset_selected(
                batch_env_ids,
                sample_flat=True, disable_motion_resample=True,
            )

            frame_limits = self.library.clip_lengths[batch_clip_ids].clamp(
                max=self.cfg.max_eval_steps
            )
            max_len = int(frame_limits.max().item())

            for step_idx in range(max_len):
                actions = self.actor_critic.act_inference(obs)
                obs, _, _, _, _ = self.env.step(actions)
                active = frame_limits > step_idx
                if not active.any():
                    continue

                # failure 语义完全由 component 是否带 threshold 决定
                for name, spec in self.cfg.evaluation_components.items():
                    val = self._compute_metric(name, batch_env_ids)
                    log_acc[name].append(val[active].detach())
                    if spec.threshold is not None:
                        fail_now = val > spec.threshold
                        motion_failed[batch_clip_ids[active]] |= fail_now[active]

        # v1: drop current training rollout, simplest env recovery
        self.env.reset_all()
        self.actor_critic.train(True)

        self._update_weights(motion_failed)
        self._dump_failed(motion_failed, current_epoch)
        self.eval_count += 1

        log_out = {"eval/success_rate": 1.0 - motion_failed.float().mean().item()}
        for name, chunks in log_acc.items():
            if chunks:
                log_out[f"eval/{name}/mean"] = torch.cat(chunks).mean().item()
        return log_out

    def _compute_metric(self, name: str, env_ids: Tensor) -> Tensor:
        """直接从 §7 reward kernel 或 §8 termination 层复用误差函数算 evaluation metric。
        failure 与否不在这里硬编码，而由 cfg.evaluation_components[name].threshold 决定。"""
        from general_tracking.tasks.general_tracking.mdp import metrics
        fn_map = {
            "anchor_ori":          metrics.anchor_ori_error_value,
            "relative_body_pos":   metrics.relative_body_pos_max_error,
            "anchor_height_error": metrics.anchor_height_error_value,
            "gt_error":            metrics.mean_body_pos_error,
            "gr_error":            metrics.mean_body_rot_error,
            "max_joint_error":     metrics.max_joint_pos_error,
        }
        return fn_map[name](self.env, command_name="motion")[env_ids]

    def _update_weights(self, motion_failed: Tensor) -> None:
        """对齐 mimic_evaluator.py:106-130"""
        success_discount = self.cfg.success_discount ** self.cfg.eval_metrics_every
        new_weights = self.library.clip_weights.clone()
        success_mask = ~motion_failed
        failed_mask = motion_failed
        new_weights[success_mask] *= success_discount
        if self.cfg.failure_discount != 0:
            new_weights[failed_mask] /= self.cfg.failure_discount
        else:
            new_weights[failed_mask] = 1.0   # 不做 sum 归一化
        self.library.update_weights(new_weights)

    def _dump_failed(self, motion_failed: Tensor, epoch: int) -> None:
        failed_ids = torch.nonzero(motion_failed).flatten().tolist()
        out_dir = self.log_dir / "failed_motions"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"failed_motions_epoch_{epoch}_rank_0.txt").write_text(
            "\n".join(str(i) for i in failed_ids)
        )
```

Runner 里的调用点（`learning/ppo/runner.py`）：

```python
class GeneralTrackingOnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(self, env, cfg, ...):
        super().__init__(env, cfg, ...)
        self.motion_evaluator = MotionSuccessEvaluator(
            env=env,
            actor_critic=self.alg.actor_critic,
            library=env.command_manager.get_term("motion").library,
            cfg=cfg.evaluator,
            log_dir=Path(self.log_dir),
        )

    def learn(self, num_iterations, ...):
        for it in range(num_iterations):
            self.collect_rollouts()
            self.alg.update()
            if it > 0 and it % self.motion_evaluator.cfg.eval_metrics_every == 0:
                log_out = self.motion_evaluator.run_eval(current_epoch=it)
                self.writer.log(log_out, step=it)
            self.save_checkpoint(it)
```

**验收提醒**（对齐 ProtoMotions recipe，与此前表述相反）：在 `mlp_bm_l2c2` 默认配置下**只有 `anchor_height_error > 0.25` 判 fail**，其他 5 个 component 在 ProtoMotions `combine_evaluation` 里因 `threshold=None` 根本不进 `failed_buf`。`test_motion_success_evaluator.py` 需测的是："当 `anchor_height_error` 未超阈但 `max_joint_error` 超自己设的阈值时，`motion_failed[i] == False`"——即 **log-only metric 不影响 fail**。想要"多指标 OR"语义需要在 `MotionSuccessEvaluatorCfg` 里显式为若干 log-only metric 也设阈值，这已偏离 ProtoMotions recipe，v1 不做。

## 13. 配置

### 13.1 G1GeneralTrackingPPOCfg（`tasks/general_tracking/config/g1/rl_cfg.py`）

```python
@dataclass
class G1GeneralTrackingPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    save_interval: int = 500
    empirical_normalization: bool = True
    experiment_name: str = "general_tracking_g1"

    obs_groups: dict = field(default_factory=lambda: {
        "actor":       ("actor_noisy",),
        "actor_clean": ("actor_clean",),
        "critic":      ("critic",),
    })

    policy: GeneralTrackingActorCriticCfg = field(default_factory=GeneralTrackingActorCriticCfg)

    algorithm: GeneralTrackingPPOCfg = field(default_factory=lambda: GeneralTrackingPPOCfg(
        actor_learning_rate=2e-5,
        critic_learning_rate=1e-4,
        num_learning_epochs=2,
        num_mini_batches=4,
        gradient_clip_val=50.0,
        clip_param=0.2,
        entropy_coef=0.0,
        value_loss_coef=1.0,
        gamma=0.99,
        lam=0.95,
        schedule="fixed",
        use_clipped_value_loss=True,
        normalize_advantages=True,
        shift_mean_advantages=True,
        l2c2_coef=1.0,
    ))

    evaluator: MotionSuccessEvaluatorCfg = field(default_factory=lambda: MotionSuccessEvaluatorCfg(
        eval_metrics_every=200,
        max_eval_steps=600,
        success_discount=0.999,
        failure_discount=0.0,
        evaluation_components={
            "anchor_ori": EvalMetricSpec(log_only=True),
            "relative_body_pos": EvalMetricSpec(log_only=True),
            "anchor_height_error": EvalMetricSpec(threshold=0.25),
            "gt_error": EvalMetricSpec(log_only=True),
            "gr_error": EvalMetricSpec(log_only=True),
            "max_joint_error": EvalMetricSpec(log_only=True),
        },
    ))

    save_predicted_motion_lib_every: int = 3
```

### 13.2 `tasks/general_tracking/gt_env_cfg.py` 与 G1 overrides

```python
# gt_env_cfg.py: robot-agnostic 通用 env cfg 工厂
def make_general_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = ManagerBasedRlEnvCfg(...)
    cfg.episode_length_s = 20.0
    cfg.commands["motion"] = MultiClipMotionCommandCfg(
        future_step_offsets=(1, 2, 4, 8),
        init_start_prob=0.2,
        ...
    )
    cfg.observations = {
        "actor_noisy": ObservationGroupCfg(
            terms={"reduced": ObservationTermCfg(func=reduced_coords_obs, noise=...),
                   "target_poses": ObservationTermCfg(func=reduced_coords_target_poses,
                                                      params={"command_name": "motion",
                                                              "offsets": (1,2,4,8)}, noise=...),
                   "action_history": ObservationTermCfg(func=processed_action_history)},
            enable_corruption=True, concatenate_terms=False,
        ),
        "actor_clean": ObservationGroupCfg(
            terms={...}, enable_corruption=False, concatenate_terms=False,
        ),
        "critic": ObservationGroupCfg(
            terms={"max_coords":   ObservationTermCfg(func=max_coords_obs),
                   "target_poses": ObservationTermCfg(func=max_coords_target_poses,
                                                      params={"command_name": "motion",
                                                              "offsets": (1,2,4,8)}),
                   "action_history": ObservationTermCfg(func=processed_action_history)},
            enable_corruption=False, concatenate_terms=True,
        ),
    }
    # rewards/terminations/events 按 §7/§8/§10 填入
    return cfg

# config/g1/env_cfgs.py: G1-specific train/play
def make_g1_train_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = make_general_tracking_env_cfg()
    cfg.scene.entities["robot"] = get_g1_robot_cfg()
    cfg.scene.num_envs = 4096
    cfg.commands["motion"].motion_library_path = "${MOTION_LIB_PATH}"
    cfg.commands["motion"].anchor_body_name = ANCHOR_BODY_NAME
    cfg.commands["motion"].body_names = BODY_NAMES
    cfg.actions["joint_pos"] = BMPositionActionCfg(
        asset_name="robot",
        joint_names=(".*",),
        scale=build_g1_bm_action_scale(),   # e/s per-joint, 非 0.25*e/s
        use_default_offset=True,
        clip=None,
    )
    return cfg
```

## 14. 任务注册

```python
# tasks/general_tracking/config/g1/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import make_g1_train_env_cfg, make_g1_play_env_cfg
from .rl_cfg import G1GeneralTrackingPPOCfg
from general_tracking.learning.ppo.runner import GeneralTrackingOnPolicyRunner

register_mjlab_task(
    task_id="GeneralTracking-Flat-Unitree-G1",
    env_cfg=make_g1_train_env_cfg(),
    play_env_cfg=make_g1_play_env_cfg(),
    rl_cfg=G1GeneralTrackingPPOCfg(),
    runner_cls=GeneralTrackingOnPolicyRunner,
)
```

## 15. 阶段划分

### 阶段 1：包骨架 + 数据契约（Day 1-3）

- T1 `bootstrap-package`：pyproject（含 `[project.scripts]` 三个入口）、`scripts/` 空壳、`src/general_tracking/` 目录树、ruff/pyright/pytest CI
- T2 `freeze-body-schema`：`robots/g1/schema.py` 写死 BODY_NAMES / ANCHOR_BODY_NAME / DENSITY_WEIGHTS，数值对比 ProtoMotions
- T3 `motion-library-core`：MotionLibrary（整数 step，manifest `control_fps` vs env fps assert）+ 单测
- T4 `manifest-builder`：`data/cli/build_manifest.py` + 单测（缺失/错误 `control_fps` 报错）
- T5 `shell-launchers`：`scripts/train.sh` / `play.sh` / `build_manifest.sh`（thin wrappers）

### 阶段 2：观测系统（Day 4-7）

- T6 `reduced-coords-obs`
- T7 `reduced-coords-target-poses`
- T8 `max-coords-obs-critic`
- T9 `max-coords-target-poses-critic`
- T10 `processed-action-history`（env-side 1-slot buffer）
- T11 `clean-noisy-obs-groups`

### 阶段 3：Commands + Rewards + Terminations + Actions（Day 7-10）

- T12 `multi-clip-command`
- T13 `bm-pd-action`
- T14 `region-density-weights` + `region-weighted-reward-kernels` + `global-anchor-ori-reward` + `reward-set-assembly`
- T15 `single-fall-termination`
- T16 `dr-events`
- T17 env build 冒烟

### 阶段 4：PPO 扩展（Day 10-14）

- T18 `actor-critic-network`
- T19 `l2c2-loss-module` + 单测
- T20 `general-tracking-ppo`
- T21 `motion-success-evaluator` + 单测
- T22 `general-tracking-runner`
- T23 `g1-env-cfg` + `g1-rl-cfg`
- T24 `task-registration`

### 阶段 5：训练验证（Day 14-18）

- T25 `smoke-training`
- T26 `small-training`
- T27 `full-training`
- T28 `play-eval`

### 阶段 6：清理（Day 19-20）

- T29 `tests-lint`
- T30 `docs`

### 阶段 7：v2（Day 21+）

- T31 `multi-gpu-v2`

## 16. 接口对齐表


| ProtoMotions 概念                                                                                                                              | general_tracking 等价物                                                                                                    | 源码定位                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `MotionLib.{gts,grs,gvs,gavs,dps,dvs}`                                                                                                       | `MotionLibrary.{body_pos_w,body_quat_w,body_lin_vel_w,body_ang_vel_w,joint_pos,joint_vel}`                              | motion_lib.py:193-525                                       |
| `MotionLib.{length_starts,motion_lengths,motion_weights,motion_dt}`                                                                          | `MotionLibrary.{clip_starts,clip_lengths,clip_weights,control_fps}`                                                     | 同上                                                          |
| `MotionLib.get_motion_state(ids, float_times)`                                                                                               | `MotionLibrary.get_state_at(clip_ids, int_time_steps)`（Option A，无插值）                                                    | motion_lib.py:317-388                                       |
| `MimicControl.future_steps=[1,2,4,8]`                                                                                                        | `MultiClipMotionCommandCfg.future_step_offsets=(1,2,4,8)`                                                               | mlp_bm_l2c2.py:105                                          |
| `MimicMotionManager.sample_time + init_start_prob=0.2`                                                                                       | `MotionLibrary.sample_init_time`                                                                                        | motion_manager.py:335-358, 409-417                          |
| `reduced_coords_obs_factory(use_noisy=True,root_height=False,root_vel=False)`                                                                | `reduced_coords_obs` + Unoise on actor_noisy group                                                                      | component_factories.py:106-149                              |
| `mimic_target_poses_reduced_coords_factory(include_dof_vel=True,include_xy_offset=False)`                                                    | `reduced_coords_target_poses(offsets=(1,2,4,8))`                                                                        | component_factories.py:336-385; obs/target_poses.py:354-540 |
| `clean_reduced_coords_*`                                                                                                                     | `actor_clean` group (enable_corruption=False)                                                                           | mlp_bm_l2c2.py:124-133                                      |
| `max_coords_obs_factory(local_obs=True,root_height=True)` + `mimic_target_poses_max_coords_factory(with_velocities=True,with_relative=True)` | `max_coords_obs` + `max_coords_target_poses` (critic group)                                                             | component_factories.py:65-103, 257-300                      |
| `previous_actions_factory(history_steps=1, processed=True)`                                                                                  | `processed_action_history` (1-slot buffer)                                                                              | component_factories.py:227-254                              |
| `L2C2Config(lambda_l2c2=1.0, obs_pairs={noisy↔clean × 2})`                                                                                   | `compute_l2c2_loss` + `GeneralTrackingPPO.update`（scalar `input_dist.detach()` / scalar `output_dist`；元素数加权）            | `agent.py:497-524`, `mlp_bm_l2c2.py:311-318`                |
| `bm_pd_action` (effort_limit/stiffness)                                                                                                      | `BMPositionAction(JointPositionAction)`；scale dict 由 `build_g1_bm_action_scale()` 从 `G1_ARTICULATION.actuators` 推 `e/s` | `action_functions.py:334-413`                               |
| `anchor_height_error_term_factory(threshold=0.25)`                                                                                           | 单 fall termination                                                                                                      | mlp_bm_l2c2.py:154                                          |
| `global_anchor_ori_rew_factory(weight=0.5, sigma=0.4)`                                                                                       | `motion_global_anchor_ori` reward                                                                                       | mlp_bm_l2c2.py:160                                          |
| `use_region_weights=True` + `compute_body_density_weights(discount=0.9)`                                                                     | `DENSITY_WEIGHTS` in schema.py + `region_weighted_*` kernels                                                            | pose_lib.py:204-262                                         |
| `MimicEvaluator.evaluate + _update_motion_sampling_weights`                                                                                  | `MotionSuccessEvaluator.run_eval + _update_weights`                                                                     | mimic_evaluator.py:78-130                                   |
| `MimicEvaluatorConfig(eval_metrics_every=200, max_eval_steps=600, success_discount=0.999, failure_discount=0)`                               | `MotionSuccessEvaluatorCfg(evaluation_components={anchor_height_error: threshold=0.25, others: log-only})`              | mlp_bm_l2c2.py:319-334                                      |
| `combine_evaluation`（只有 `threshold != None` 的 component 进 `failed_buf`）                                                                      | `MotionSuccessEvaluator` 仅带 threshold 的 component 触发 fail；默认只有 `anchor_height_error > 0.25` 改权重，其余 metrics 只记录          | `envs/base_env/utils.py:209-217`                            |
| `PPOAgentConfig(...)`                                                                                                                        | `GeneralTrackingPPOCfg`                                                                                                 | mlp_bm_l2c2.py:297-337                                      |
| Actor 6×1024 ReLU / Critic 4×1024 ReLU + actor_logstd=-2.9 learnable + normalize_obs(clamp=5)                                                | `GeneralTrackingActorCritic` + `empirical_normalization=True`                                                           | mlp_bm_l2c2.py:241-277                                      |
| `examples/experiments/mimic/mlp_bm_l2c2.py`                                                                                                  | `tasks/general_tracking/config/g1/env_cfgs.py` + `rl_cfg.py`                                                            | -                                                           |


## 17. 验收标准（v1）

1. `uv run list-envs` 列出 `GeneralTracking-Flat-Unitree-G1`；`uv run gt-train --help` / `gt-play --help` / `gt-build-manifest --help` 都能拉起。
2. `pytest tests/` 全通。关键回归项：
  - `test_l2c2_loss.py`：λ=0 时 total_loss 数值等价 baseline
  - `test_region_density_weights.py`：`DENSITY_WEIGHTS` 与 ProtoMotions `compute_body_density_weights` 在 G1 kinematic tree 上逐元素 diff < 1e-5
  - `test_motion_success_evaluator.py`：2-clip 库 + mock policy（clip 0 保持稳定、clip 1 `anchor_height_error > 0.25`），一次 `run_eval` 后 `w[1]==1.0, w[0] ≈ 0.819`，`failed_motions_epoch_N.txt` 内容为 `[1]`；额外用例：当 `anchor_height_error` 始终 < 0.25 但 `max_joint_error` 很大时，`**motion_failed[i] == False**`（即 log-only metric 不影响 fail，对齐 ProtoMotions `combine_evaluation` 只看带 threshold 的 component）
  - `test_motion_library.py`：manifest `control_fps != env_control_fps` 立即报错
3. `bash scripts/train.sh NUM_ENVS=4096 MAX_ITER=10000` 完成，训练日志曲线：
  - `motion_relative_body_pos` reward 稳定上升
  - `motion_global_anchor_ori` reward 稳定上升
  - `loss/l2c2` 非零、稳定
  - iter ≥ 200 后可见第一次 evaluator 触发，`motion_weights` 熵下降
4. `bash scripts/play.sh CKPT=<path>` 加载 + ghost 回放，可视化 target motion 与 robot 同步。

## 18. 记账：已彻底定死、不再"待定"


| 参数                         | 值                                                                                                          | 来源                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `lambda_l2c2`              | `1.0`                                                                                                      | `mlp_bm_l2c2.py:313`                          |
| L2C2 公式                    | `input_dist=(Σdiff².sum()/Σnumel).detach()`、`output_dist=(mu_n-mu_c)².mean()`、`λ·output/(input+1e-8)`      | `agent.py:497-524`                            |
| `actor_logstd (init)`      | `-2.9` (learnable)                                                                                         | `:243-244`                                    |
| `actor_lr, critic_lr`      | `2e-5`, `1e-4`                                                                                             | `:298, 301`                                   |
| Adam betas                 | `(0.95, 0.99)`                                                                                             | `:298, 301`                                   |
| `num_mini_epochs`          | `2`                                                                                                        | `:307`                                        |
| `gradient_clip_val`        | `50.0`                                                                                                     | `:309`                                        |
| `entropy_coef`             | `0.0`                                                                                                      | implicit default                              |
| `clip_critic_loss`         | `True`                                                                                                     | `:310`                                        |
| `adaptive_lr`              | `False`                                                                                                    | `:305`                                        |
| `normalize_rewards`        | `False`                                                                                                    | `:304`                                        |
| `advantage_normalization`  | `enabled=True, shift_mean=True`                                                                            | `:335-337`                                    |
| `normalize_obs, clamp`     | `True, 5`                                                                                                  | `:257-258, 273-274`                           |
| `eval_metrics_every`       | `200`                                                                                                      | `evaluators/config.py:38`                     |
| `max_eval_steps`           | `600`                                                                                                      | `evaluators/config.py:34`                     |
| `success_discount`         | `0.999`                                                                                                    | `:331`                                        |
| `failure_discount`         | `0`                                                                                                        | `:332`                                        |
| `init_start_prob`          | `0.2`                                                                                                      | `:207`                                        |
| `max_episode_length`       | `1000` steps (20 s)                                                                                        | `:199`                                        |
| `future_steps`             | `[1, 2, 4, 8]`                                                                                             | `:105`                                        |
| `fall threshold`           | `0.25 m`                                                                                                   | `:154`                                        |
| `history_steps, processed` | `1, True`                                                                                                  | `:147-149, 200`                               |
| `BMPositionAction.scale`   | 从 `G1_ARTICULATION.actuators` 解析 `effort_limit / stiffness`（无 0.25）                                        | `action_functions.py:403`                     |
| `region_weights`           | `compute_body_density_weights(discount=0.9)`                                                               | `pose_lib.py:204-262`                         |
| `anchor_body_name`         | `torso_link`                                                                                               | `mlp_bm_l2c2.py:348`                          |
| Actor 架构                   | `MLP 6×1024 ReLU`                                                                                          | `:261`                                        |
| Critic 架构                  | `MLP 4×1024 ReLU`                                                                                          | `:276`                                        |
| Time indexing              | Option A：整数 step；manifest `control_fps` vs env control fps 强校验（NPZ 不塞 fps）                                 | 本项目决策                                         |
| Evaluator fail 判据          | 单指标：`anchor_height_error > 0.25`；其他 5 个 metric `threshold=None` → log-only                                 | `utils.py:209-217` + `mlp_bm_l2c2.py:319-334` |
| Evaluator env 状态           | v1：扫描后 `env.reset_all()` 放弃当前 rollout；不实现 save/restore_state                                               | 本项目决策                                         |
| Action owner / history     | `BMPositionAction(JointPositionAction)._history`；obs 读 `env.action_manager.get_term("joint_pos")._history` | 本项目决策                                         |
| 模块命名                       | `learning/ppo/`（非 `agents/` 非 `rl/`）                                                                       | 本项目决策                                         |
| 包/仓库名                      | `general_tracking`                                                                                         | 本项目决策                                         |
| 任务目录名                      | `tasks/general_tracking/`                                                                                  | 本项目决策                                         |
| Env cfg 文件                 | `gt_env_cfg.py`                                                                                            | 本项目决策                                         |
| Task ID                    | `GeneralTracking-Flat-Unitree-G1`                                                                          | 本项目决策                                         |
| 用户入口                       | `scripts/{train,play,build_manifest}.sh` → `[project.scripts]` gt-train/gt-play/gt-build-manifest          | 本项目决策                                         |


v1 明确 **不启用**：contact reward、`ref_contact_smooth_window`、`eval_action_ema_alpha`、`exclude_motion_ids`、`subset_method`、ONNX 导出、真机部署、多 GPU 分片（v2）。