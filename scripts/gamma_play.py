#!/usr/bin/env python3
"""
Drive the GAMMA pick task (B1 + Z1 robot) with the keyboard and watch it live,
e.g. in a NoMachine session. There is no policy and no training: actions come
from the keyboard, while observations, GSNet grasp prediction and online grasp
fusion run exactly as they do during training.

Run from anywhere (the script switches to the repo root, because the config and
GraspSensor use paths relative to it):

    python scripts/gamma_play.py
    python scripts/gamma_play.py habitat.dataset.split=val

Controls (hold the key):
    I / K           base forward / backward
    J / L           base turn left / right
    U / O           base strafe left / right
    Q W E R T Y     arm joints 1-6 +   (layout as in habitat-lab's examples/interactive_play.py)
    1 2 3 4 5 6     arm joints 1-6 -
    SPACE           toggle gripper (gaze grasp / release)
    M               new episode
    P               print base / end-effector pose and grasp counters
    ESC             quit

Window: third-person view | head RGB over arm RGB. The fused grasp poses kept by
the online fusion module are drawn on the camera images (red = low score,
green = high score); the target object is the yellow cross.
"""

import argparse
import os

import cv2
import magnum as mn
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = "rearrange/rearrange_w_grasping.yaml"
ARM_ACTION = "arm_action"
BASE_ACTION = "base_velocity_non_cylinder"
ARM_PLUS_KEYS = "qwerty"
ARM_MINUS_KEYS = "123456"
VIEW_CAMS = ("head_rgb", "articulated_agent_arm_rgb")
GRASP_NUM = 64  # grasp_sensor rows 0..63 are grasps, row 64 holds counters


def make_config(args):
    from habitat.config import read_write
    from habitat.config.default import get_agent_config
    from habitat.config.default_structured_configs import ThirdRGBSensorConfig
    from habitat_baselines.config.default import get_config

    config = get_config(args.cfg, args.opts)
    with read_write(config):
        # Background rendering needs a second GL context in another thread,
        # which the GLX (display) build of habitat-sim refuses (BadAccess).
        config.habitat.simulator.concur_render = False
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {
                "third_rgb_sensor": ThirdRGBSensorConfig(
                    height=args.cam_res, width=args.cam_res
                )
            }
        )
        if not args.keep_limits:
            # Manual driving is slow and bumps into things: don't end the
            # episode after 300 steps or on accumulated collision force.
            config.habitat.environment.max_episode_steps = 0
            force_terminate = config.habitat.task.measurements.force_terminate
            force_terminate.max_accum_force = -1.0
            force_terminate.max_instant_force = -1.0
    return config


def camera_hfovs(config):
    from habitat.config.default import get_agent_config

    agent_config = get_agent_config(sim_config=config.habitat.simulator)
    return {
        s.uuid: float(s.hfov)
        for s in agent_config.sim_sensors.values()
        if s.uuid in VIEW_CAMS
    }


def project(points, sim, uuid, hfov_deg, shape):
    """World points -> pixel coords in a pinhole camera (None if behind it)."""
    node = sim._sensors[uuid]._sensor_object.node
    cam_from_world = node.absolute_transformation().inverted()
    h, w = shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    pixels = []
    for p in points:
        pc = cam_from_world.transform_point(mn.Vector3(*p))
        # Habitat cameras look along -Z, with +Y up.
        if pc[2] > -1e-3:
            pixels.append(None)
            continue
        pixels.append(
            (int(w / 2.0 + f * pc[0] / -pc[2]), int(h / 2.0 - f * pc[1] / -pc[2]))
        )
    return pixels


def draw_overlays(images, env, hfovs):
    grasps = env.task.sensor_suite.sensors[
        "grasp_sensor"
    ].fusion_grasp_gltree.all_grasps_array()
    target = np.asarray(env.sim.get_target_objs_start())
    max_score = max(grasps[:, 12].max(), 1e-6) if len(grasps) else 1.0
    for uuid, img in images.items():
        for i, px in enumerate(
            project(grasps[:, :3], env.sim, uuid, hfovs[uuid], img.shape)
        ):
            if px is not None:
                s = min(grasps[i, 12] / max_score, 1.0)
                cv2.circle(img, px, 3, (int(255 * (1 - s)), int(255 * s), 0), -1)
        for px in project(target, env.sim, uuid, hfovs[uuid], img.shape):
            if px is not None:
                cv2.drawMarker(img, px, (255, 255, 0), cv2.MARKER_CROSS, 12, 2)


def compose(obs, images, lines):
    third = np.ascontiguousarray(obs["third_rgb"][..., :3])
    h = third.shape[0]
    right = np.concatenate([images[k] for k in VIEW_CAMS], axis=0)
    right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    frame = np.ascontiguousarray(np.concatenate([third, right], axis=1))
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
        )
    return frame


def read_held_keys():
    keys = pygame.key.get_pressed()
    base = np.zeros(3, dtype=np.float32)  # [forward, lateral (+left), turn (+left)]
    if keys[pygame.K_i]:
        base[0] = 1.0
    elif keys[pygame.K_k]:
        base[0] = -1.0
    if keys[pygame.K_u]:
        base[1] = 1.0
    elif keys[pygame.K_o]:
        base[1] = -1.0
    if keys[pygame.K_j]:
        base[2] = 1.0
    elif keys[pygame.K_l]:
        base[2] = -1.0

    arm = np.zeros(len(ARM_PLUS_KEYS), dtype=np.float32)
    for i, (plus, minus) in enumerate(zip(ARM_PLUS_KEYS, ARM_MINUS_KEYS)):
        if keys[getattr(pygame, f"K_{plus}")]:
            arm[i] = 1.0
        elif keys[getattr(pygame, f"K_{minus}")]:
            arm[i] = -1.0
    return base, arm


def print_state(env, obs):
    agent = env.sim.articulated_agent
    counters = np.asarray(obs["grasp_sensor"])[GRASP_NUM]
    print(
        f"base pos {np.round(np.asarray(agent.base_pos), 3)}  "
        f"ee pos {np.round(np.asarray(agent.ee_transform().translation), 3)}\n"
        f"arm joints {np.round(np.asarray(agent.arm_joint_pos), 3)}\n"
        f"fused grasps {int(counters[2])}  edges {int(counters[3])}  "
        f"fused count {int(counters[4])}  total score {counters[5]:.2f}"
    )


def play(env, hfovs, fps):
    obs = env.reset()
    # Start pygame only after Habitat has created its OpenGL contexts: doing it
    # before makes both fight over the X display (GLXMakeCurrent BadAccess).
    pygame.init()
    grip_cmd = -1.0  # GazeGraspAction: >= 0 tries to grasp, < 0 releases
    total_reward, step = 0.0, 0
    screen, clock = None, pygame.time.Clock()

    while True:
        new_episode = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grip_cmd = -grip_cmd
                elif event.key == pygame.K_m:
                    new_episode = True
                elif event.key == pygame.K_p:
                    print_state(env, obs)

        if new_episode or env.episode_over:
            print(f"episode over after {step} steps, return {total_reward:+.2f}")
            obs = env.reset()
            grip_cmd, total_reward, step = -1.0, 0.0, 0

        base, arm = read_held_keys()
        obs = env.step(
            {
                "action": (ARM_ACTION, BASE_ACTION),
                "action_args": {
                    "arm_action": arm,
                    "grip_action": np.array([grip_cmd], dtype=np.float32),
                    "base_vel": base,
                },
            }
        )
        metrics = env.get_metrics()
        reward = float(metrics.get("pick_reward", 0.0))
        total_reward += reward
        step += 1

        counters = np.asarray(obs["grasp_sensor"])[GRASP_NUM]
        images = {k: np.ascontiguousarray(obs[k][..., :3]) for k in VIEW_CAMS}
        draw_overlays(images, env, hfovs)
        frame = compose(
            obs,
            images,
            [
                f"ep {env.current_episode.episode_id}  step {step}  "
                f"r {reward:+.3f}  R {total_reward:+.2f}",
                f"fused grasps {int(counters[2])}  edges {int(counters[3])}",
                f"gripper {'GRASP' if grip_cmd >= 0 else 'open'}  "
                f"holding {int(np.asarray(obs['is_holding']).item())}  "
                f"success {int(bool(metrics.get('pick_success', False)))}",
            ],
        )

        if screen is None:
            screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption("GAMMA play")
        screen.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
        pygame.display.update()
        clock.tick(fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--cam-res", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--keep-limits",
        action="store_true",
        help="keep the training step limit (300) and force termination",
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="config overrides, e.g. habitat.dataset.split=val"
    )
    args = parser.parse_args()
    if pygame is None:
        raise ImportError("Need PyGame: python -m pip install pygame")

    os.chdir(REPO_ROOT)
    import habitat

    config = make_config(args)
    with habitat.Env(config=config) as env:
        play(env, camera_hfovs(config), args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
