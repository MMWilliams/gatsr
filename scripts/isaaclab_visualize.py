"""Visualize the Unitree G1 in the Isaac Sim viewport while a GATS-R method
drives it.

Unlike ``isaaclab_smoke.py`` / ``isaaclab_benchmark.py`` (which force headless),
this script leaves rendering ON so the Isaac Sim window opens and you can watch
the robot. Use a small ``--num_envs`` (1-4) for a clean view.

Methods:
    random        - uniform actions (robot flails / falls; good sanity check)
    mppi          - batched MPPI in the L2 latent model
    gatsr_full    - full GATS-R: MPPI + CBF + monitor + graph-indexed recovery
    zero          - hold default pose (no policy); just look at the scene

Run (from the repo root, in the isaaclab conda env via the launcher):

    pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_visualize.py `
        --task Isaac-Velocity-Rough-G1-v0 --num_envs 2 --method gatsr_full `
        --train_steps 512 --run_steps 2000

Controls in the viewport: left-drag orbit, scroll zoom, `F` to frame the
selection, `Space` to pause physics. Close the window (or Ctrl-C in the
terminal) to exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

parser = argparse.ArgumentParser(description="GATS-R G1 visualizer (windowed).")
parser.add_argument("--task", default="Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--method", default="gatsr_full",
                    choices=["zero", "random", "mppi", "gatsr_full"])
parser.add_argument("--train_steps", type=int, default=512,
                    help="random transitions used to fit the L2 model (methods that need it)")
parser.add_argument("--run_steps", type=int, default=2000,
                    help="number of control steps to visualize")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# KEY DIFFERENCE vs smoke/benchmark: do NOT force headless -> a window opens.
args_cli.headless = False
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from gatsr.isaaclab.env import IsaacLabG1Env, IsaacLabG1Config  # noqa: E402
from gatsr.isaaclab.latent import G1EnsembleLatentModel, G1LatentConfig  # noqa: E402
from gatsr.isaaclab.agent import G1Agent, G1AgentConfig  # noqa: E402


def collect_random_data(env: IsaacLabG1Env, n_steps: int):
    s, a, sp = [], [], []
    env.reset(seed=0)
    for _ in range(n_steps):
        ps = env.physical_state.clone()
        act = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
        env.step(act)
        s.append(ps)
        a.append(act)
        sp.append(env.physical_state.clone())
    return torch.cat(s), torch.cat(a), torch.cat(sp)


def make_cost_fn():
    target_vx = 0.5

    def cost_fn(traj, actions, eps):
        lin_vel_x = traj[..., 0]
        tilt = traj[..., 6:8].norm(dim=-1)
        vel_err = (lin_vel_x.mean(dim=-1) - target_vx).abs()
        upright = tilt.mean(dim=-1)
        action_cost = actions.pow(2).mean(dim=(-1, -2))
        return vel_err + 1.5 * upright + 0.005 * action_cost + 0.05 * eps.mean(dim=-1)

    return cost_fn


def main() -> int:
    print(f"[viz] launching {args_cli.task} with {args_cli.num_envs} envs (windowed) ...", flush=True)
    env = IsaacLabG1Env(IsaacLabG1Config(
        task=args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device, seed=0,
    ))
    print(f"[viz] obs_dim={env.obs_dim} action_dim={env.action_dim} "
          f"physical_dim={env.physical_dim}", flush=True)

    agent = None
    cost_fn = None
    if args_cli.method in ("mppi", "gatsr_full"):
        print(f"[viz] collecting {args_cli.train_steps} random transitions to fit L2 ...", flush=True)
        S, A, SP = collect_random_data(env, args_cli.train_steps)
        model = G1EnsembleLatentModel(G1LatentConfig(
            state_dim=env.physical_dim, action_dim=env.action_dim,
            device=str(env.device), multi_gpu_rollouts=False,
        ))
        model.fit(S, A, SP, verbose=False)
        print("[viz] L2 trained.", flush=True)
        agent = G1Agent(env, model, G1AgentConfig(
            use_mppi=True,
            use_cbf=args_cli.method == "gatsr_full",
            use_monitor=args_cli.method == "gatsr_full",
            use_recovery=args_cli.method == "gatsr_full",
            horizon=6, n_samples=32, n_iter=1,
        ))
        cost_fn = make_cost_fn()

    env.reset(seed=0)
    print(f"[viz] running {args_cli.run_steps} steps with method={args_cli.method}. "
          f"Close the window or Ctrl-C to stop.", flush=True)
    t0 = time.perf_counter()
    step = 0
    while simulation_app.is_running() and step < args_cli.run_steps:
        with torch.inference_mode():
            ps = env.physical_state
            if args_cli.method == "zero":
                actions = torch.zeros(env.num_envs, env.action_dim, device=env.device)
            elif args_cli.method == "random":
                actions = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
            else:
                actions, info = agent.act(ps, cost_fn)
            rec = None
            if agent is not None and args_cli.method == "gatsr_full":
                rec = info["recovery_mask"]
            if rec is not None and bool(rec.any()):
                env.recover_step(actions)
            else:
                env.step(actions)
        step += 1
        if step % 100 == 0:
            fps = step / (time.perf_counter() - t0)
            extra = ""
            if agent is not None:
                extra = (f" cbf_viol={int(agent.safety_violations.sum())} "
                         f"ood={int(agent.failures_detected.sum())} "
                         f"rec_att={int(agent.recovery.attempts.sum())} "
                         f"rec_ok={int(agent.recovery.successes.sum())}")
            print(f"[viz] step {step}/{args_cli.run_steps}  {fps:.0f} ctrl-steps/s{extra}", flush=True)

    print("[viz] done.", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
