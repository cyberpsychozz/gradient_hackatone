# Repository Map: Mars Rover Public Environment

## 📁 Root Directory
- `docs/Agent.md` - High-level overview for AI agents.
- `environment/` - The primary codebase containing the engine and environment.
- `docs/` - Documentation, working rules, and biome profiles.

---

## 📁 `environment/` (Core Repository)

### 🛠️ Engine & Build
- `cpp/` - The high-performance C++ source.
    - `include/mars/` - Header files for the core engine.
        - `biomes/` - Biome-specific logic and templates.
        - `action.hpp`, `physics.hpp`, `reward.hpp`, etc. - Core engine interfaces.
    - `src/` - Implementation files.
        - `physics/` - Low-level physics step logic (contacts, dynamics, etc.).
        - `terrain.cpp`, `mechanics.cpp`, `renderer.cpp` - Core logic components.
    - `bindings/` - Pybind11 glue code for Python integration.
    - `CMakeLists.txt` - Build configuration.
- `python/` - Python package structure.
    - `mars_rover_env/` - The Gymnasium/Gym-compatible wrapper.
        - `envs/` - Environment implementations (`mars_rover_env.py`, `mars_rover_vec_env.py`).
        - `configs/` - YAML configuration files for env/rover settings.
        - `actions.py`, `config.py`, `fingerprint.py` - Supporting logic.
- `build/` - Compiled artifacts (generated during build).
- `Makefile` - Build automation (check, submission).
- `setup.py` / `pyproject.toml` - Python packaging.
- `train.py` - Script for training agents.
- `model.py` - Model definitions/utilities.

### 🖥️ GUI (Visualization)
- `gui/` - The interactive visualizer.
    - `python/mars_rover_gui/` - GUI application logic (app, controls, HUD).
    - `tests/` - GUI-specific tests.
    - `pyproject.toml` / `Makefile` - GUI packaging and build.

### 🧪 Testing & Quality
- `tests/` - Core environment tests (smoke tests).
- `check_policy.py` - Policy compliance checker.
- `package_submission.py` - Submission preparation script.

### 🖼️ Assets
- `images/` - Sample images/reference photos.

---

## 🛠️ Development & Workflow Summary

### 1. Setup & Installation
1. Install dependencies: `pip install setuptools wheel pybind11`
2. Build & Install: `pip install --no-build-isolation --force-reinstall .`
3. Install GUI: `pip install --no-build-isolation --force-reinstall ./gui`

### 2. Common Commands
- `make check` - Verify environment/contract integrity.
- `make submission` - Package the environment for evaluation.
- `mars-rover-play --list-biomes` - List available biomes in GUI.

### 3. Extending the Environment
- **New Biome:** Edit `environment/cpp/include/mars/custom_biomes.inc.hpp`.
- **New Mechanics:** Modify `environment/cpp/src/mechanics.cpp` or `physics/`.
- **New Configuration:** Update YAMLs in `environment/python/mars_rover_env/configs/`.
