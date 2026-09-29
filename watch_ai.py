"""Watch the trained rover policy drive (and take over with keyboard if needed).

Usage:
    python watch_ai.py                       # AI mode, training config
    python watch_ai.py --biome 13            # fixed biome world
    python watch_ai.py --ckpt artifacts/best.pt --fps 30

Keys (AI mode): Esc/Q quit · R new world · T pause/resume.
While paused or holding any control key (arrows/WASD/J/L/K... see GUI), the
human takes over; releasing controls returns to the policy.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ENV_PY = ROOT / "environment" / "python"
MODEL_DIR = ROOT / "solutions" / "submission"
for path in (ENV_PY, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from mars_rover_env import MarsRoverEnv  # noqa: E402
from mars_rover_env.actions import ACTION_MACROS  # noqa: E402
from model import Policy  # noqa: E402

try:
    from mars_rover_gui.hud import telemetry_lines
    from mars_rover_gui.controls import keyboard_action
except ImportError:
    telemetry_lines = None
    keyboard_action = lambda keys: 0


def load_policy(ckpt: str, hidden: int) -> Policy:
    policy = Policy(160, len(ACTION_MACROS), hidden)
    state = torch.load(ckpt, map_location="cpu")
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    policy.load_state_dict(state)
    policy.eval()
    return policy


class AiPlayer:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk

        self.args = args
        self.env = MarsRoverEnv(
            config_path=args.config,
            biome_split=1,
            fixed_biome_id=args.biome,
            render_mode="debug_rgb_array" if args.debug else "rgb_array",
            render_width=args.width,
            render_height=args.height,
        )
        self.seed = args.seed
        self.obs, _ = self.env.reset(seed=self.seed, options={"trial_start": True})
        self.frame_ms = max(1, round(1000 / max(1, args.fps)))
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.keys: set[str] = set()
        self.paused = False
        self.steps_in_action = 0
        self.phys_steps = 0
        self.action_index = 0
        self.action_bitmask = 0
        self.prev_action = torch.zeros(1, dtype=torch.long)
        self.prev_reward = torch.zeros(1)
        self.prev_done = torch.zeros(1)
        self.trial_start = torch.ones(1)
        self.memory = torch.zeros(1, args.hidden)

        self.root = tk.Tk()
        self.root.title("Mars Rover — AI watch")
        self.fullscreen = bool(args.fullscreen)
        self.sidebar_width = 460
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
            args.width = max(640, self.root.winfo_screenwidth() - self.sidebar_width - 16)
            args.height = max(360, self.root.winfo_screenheight() - 16)
        self.canvas = tk.Label(self.root, bg="#000000", borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.sidebar = tk.Frame(self.root, width=self.sidebar_width, bg="#171717", padx=10, pady=10)
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(False)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        tk.Label(self.sidebar, text="AI TELEMETRY", bg="#171717", fg="#ffffff",
                 font=("Consolas", 13, "bold")).pack(anchor="w", pady=(0, 6))
        self.telemetry_label = tk.Label(self.sidebar, justify="left", anchor="nw",
                                        bg="#171717", fg="#f4f4f4", font=("Consolas", 9, "bold"))
        self.telemetry_label.pack(fill="x")
        tk.Label(self.sidebar, text="KEYS: Esc quit · R new world · T pause · arrows/WASD take over",
                 bg="#171717", fg="#888888", font=("Consolas", 8), wraplength=self.sidebar_width - 20,
                 justify="left").pack(anchor="w", pady=(8, 0))
        self.status = tk.Label(self.sidebar, justify="left", anchor="sw",
                               wraplength=self.sidebar_width - 20, bg="#171717",
                               fg="#aaaaaa", font=("Consolas", 8))
        self.status.pack(side="bottom", fill="x")
        self.photo = None
        self.root.bind_all("<KeyPress>", self._press)
        self.root.bind_all("<KeyRelease>", self._release)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after_idle(self.root.focus_force)
        self.root.after(0, self._tick)

    def _press(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys:
            return
        self.keys.add(key)
        if key in {"escape", "q"}:
            self._close()
        elif key == "r":
            self._new_world()
        elif key == "t":
            self.paused = not self.paused
        elif key == "f11":
            self.fullscreen = not self.fullscreen
            self.root.attributes("-fullscreen", self.fullscreen)

    def _release(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def _new_world(self) -> None:
        self.seed = random.randrange(2**31)
        self.obs, _ = self.env.reset(seed=self.seed, options={"trial_start": True})
        self.memory.zero_()
        self.prev_action.zero_()
        self.prev_reward.zero_()
        self.prev_done.zero_()
        self.trial_start = torch.ones(1)
        self.steps_in_action = 0
        self.phys_steps = 0
        self.paused = False
        self.status.config(text=f"NEW WORLD seed={self.seed}")

    def _close(self) -> None:
        self.root.destroy()

    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        if elapsed > 0.0:
            instant = 1.0 / elapsed
            self.fps = instant if self.fps == 0.0 else self.fps * 0.9 + instant * 0.1

        if not self.paused:
            manual = keyboard_action(self.keys)
            if manual:
                self.action_bitmask = manual
                self.steps_in_action = 0
            elif self.steps_in_action == 0:
                obs = torch.from_numpy(self.obs.copy()).unsqueeze(0)
                progress = torch.full((1,), min(self.phys_steps / 18000.0, 1.0))
                with torch.no_grad():
                    logits, next_memory = self.policy(
                        obs, self.prev_action, self.prev_reward, self.prev_done,
                        progress, self.trial_start, self.memory,
                    )
                self.memory = next_memory
                self.action_index = int(logits.argmax(-1).item())
                self.action_bitmask = int(ACTION_MACROS[self.action_index])
                self.trial_start = torch.zeros(1)
            obs, reward, terminated, truncated, _ = self.env.step(self.action_bitmask)
            self.obs = obs
            self.steps_in_action += 1
            self.phys_steps += 1
            if self.steps_in_action >= self.args.frame_skip:
                self.steps_in_action = 0
                self.prev_action = torch.tensor([self.action_index], dtype=torch.long)
                self.prev_reward = torch.tensor([reward])
                self.prev_done = torch.tensor([float(terminated or truncated)])
            if terminated or truncated:
                debug = self.env.debug_info()
                reason = debug.get("termination_reason", 0)
                self.status.config(text=f"RUN ENDED reason={reason}  R — new world")
                self.paused = True
                self.root.after(1500, self._new_world)

        debug = self.env.debug_info()
        frame = self.env.render()
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.configure(image=self.photo)
        lines = telemetry_lines(debug, self.fps, self.seed) if telemetry_lines else str(debug)
        self.telemetry_label.config(text=lines)
        self.root.after(self.frame_ms, self._tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_ckpt = MODEL_DIR / "artifacts" / "best.pt"
    parser.add_argument("--ckpt", type=str, default=str(default_ckpt))
    parser.add_argument("--config", type=str,
                        default=str(ENV_PY / "mars_rover_env" / "configs" / "env.yaml"))
    parser.add_argument("--biome", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=45)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    args.policy = load_policy(args.ckpt, args.hidden)
    print(f"loaded policy from {args.ckpt}")
    print(f"config: {args.config} | seed: {args.seed} | biome: {args.biome}")
    player = AiPlayer(args)
    player.policy = args.policy
    player.root.mainloop()


if __name__ == "__main__":
    main()
