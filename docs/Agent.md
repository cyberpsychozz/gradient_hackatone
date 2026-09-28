# Agent.md

## Overview
Before changing training or simulation, read [working_rules.md](working_rules.md) and [run_rules.md](run_rules.md). World generation and biome profiles are in [worlds.md](worlds.md).

This repository contains the "Mars Rover Public Environment," a comprehensive development suite for reinforcement learning. It includes a high-performance C++ physics engine, Python bindings (via pybind11), a Gymnasium-compatible environment, a GUI for visualization, and tools for package submission and biome testing.

## Core Components
- **Core Engine (C++):** Located in `environment/cpp/`. Handles physics, mechanics, terrain generation, and reward calculation.
- **Python Bindings & Env:** Located in `environment/python/`. Provides `mars_rover_env` for easy integration with RL frameworks like Stable Baselines3 or Ray RLLib.
- **GUI:** Located in `environment/gui/`. A tool for visual inspection of biomes and rover behavior.
- **Build System:** Uses `Makefile` and `setup.py` to compile C++ extensions and install Python packages.

## Key Workflows
- **Installation:** `pip install --no-build-isolation --force-reinstall .`
- **Biome Creation:** Modify `environment/cpp/include/mars/custom_biomes.inc.hpp`.
- **Validation:** Run `make check` to verify environment contracts.
- **Submission:** Run `make submission` to prepare the package for the evaluator.

## Development Guidelines
- **Performance:** Keep core loops in C++.
- **Contract Integrity:** Do not change action spaces, observation shapes, or rover configurations if compatibility with the official evaluator is required.
- **Testing:** Use the smoke tests in `environment/tests/` to ensure basic functionality.
