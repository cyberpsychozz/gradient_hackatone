"""Balanced, reproducible terrain and mechanic profiles for PPO training.

Each profile is a *distribution*: reset seeds still change crater positions,
biome order, terrain phases and mechanic parameters. The eval_stress YAMLs
are kept outside this training mixture as local holdouts.
"""
from __future__ import annotations

from dataclasses import dataclass

from _mars_rover_cpp import MechanicType
from mars_rover_env.config import load_env_config


@dataclass(frozen=True)
class WorldProfile:
    name: str
    tier: str
    weight: int
    terrain_m: int = 0  # zero means use the reference environment unchanged
    craters: int = 0
    steps: int = 0
    amplitude: float = 0.82
    roughness: float = 0.62
    terrain_offset_m: float = 0.0
    biome_ramp_m: float = 0.0
    biome_zones: int = 14
    zone_min_m: float = 30.0
    zone_max_m: float = 70.0
    surprise: float = 0.4
    surprise_strength: float = 1.0
    stack: tuple[str, ...] = ()
    full_difficulty: bool = False
    initial_energy: float = 100.0
    initial_temperature: float = 20.0
    fixed_biome_id: int = -1


PROFILES = (
    # Reference-like worlds provide a stable curriculum and keep hard worlds
    # from dominating PPO's returns and early terminations.
    WorldProfile("reference", "accessible", 8),
    WorldProfile("rolling_hills", "accessible", 6, 6000, 45, 18, .72, .52,
                 600, 1600, 12, 55, 110, .18),
    WorldProfile("short_biomes", "accessible", 5, 5000, 48, 20, .76, .56,
                 700, 550, 22, 22, 48, .38),
    WorldProfile("sparse_ridges", "accessible", 5, 5500, 55, 25, .85, .63,
                 900, 1300, 16, 35, 75, .32),
    WorldProfile("cool_low_energy", "accessible", 4, 5500, 44, 18, .72, .55,
                 700, 1100, 15, 35, 80, .25,
                 initial_energy=90, initial_temperature=-5),
    WorldProfile("gentle_dunes", "accessible", 3, 5500, 38, 17, .70, .52,
                 600, 1000, 15, 35, 75, .20, stack=("Sand",), fixed_biome_id=22),
    WorldProfile("shallow_frost", "accessible", 3, 5500, 40, 18, .72, .54,
                 650, 1050, 15, 35, 75, .24, stack=("Ice",), fixed_biome_id=24),
    WorldProfile("morning_crosswind", "accessible", 3, 5500, 40, 18, .72, .54,
                 650, 1050, 15, 35, 75, .24, stack=("Wind",), fixed_biome_id=23),
    # Denser terrain and distinct traction/weather mechanisms, but with a
    # gradual biome ramp. A single named stack is intentionally repeated
    # across different random biome and terrain seeds.
    WorldProfile("broken_plain", "challenging", 6, 4000, 76, 34, .91, .69,
                 1500, 800, 18, 30, 65, .58),
    WorldProfile("sand_mud_wind", "challenging", 5, 4500, 58, 28, .78, .59,
                 1550, 1050, 17, 30, 65, .42, stack=("Sand", "Mud", "Wind")),
    WorldProfile("ice_crosswind", "challenging", 5, 4500, 60, 27, .79, .60,
                 1550, 1000, 17, 30, 65, .42, stack=("Ice", "Wind")),
    WorldProfile("viscous_mud", "challenging", 5, 4500, 58, 27, .76, .57,
                 1400, 850, 18, 25, 60, .48, stack=("Mud",)),
    WorldProfile("low_gravity_gaps", "challenging", 4, 4000, 72, 33, .87, .67,
                 1750, 850, 18, 28, 60, .55, stack=("LowGravity",)),
    WorldProfile("soft_swamp", "challenging", 4, 4500, 48, 20, .72, .56,
                 1100, 1050, 16, 30, 70, .36, stack=("Mud",), fixed_biome_id=30),
    WorldProfile("muddy_crosswind", "challenging", 3, 4500, 54, 23, .74, .57,
                 1200, 1000, 16, 30, 70, .40,
                 stack=("Mud", "Wind"), fixed_biome_id=3),
    WorldProfile("dry_heat", "challenging", 4, 4500, 48, 20, .73, .54,
                 1050, 1000, 16, 30, 70, .34, stack=("Crust",), fixed_biome_id=11,
                 initial_temperature=45),
    WorldProfile("water_basin", "challenging", 4, 4500, 42, 17, .68, .51,
                 900, 1000, 14, 35, 80, .25, stack=("Liquid",), fixed_biome_id=7),
    WorldProfile("ice_ridge", "challenging", 3, 4500, 55, 25, .80, .61,
                 1400, 950, 17, 30, 70, .42, stack=("Ice",), fixed_biome_id=29),
    WorldProfile("gust_corridor", "challenging", 3, 4500, 55, 25, .79, .60,
                 1450, 900, 17, 30, 70, .45, stack=("Wind",), fixed_biome_id=28),
    # Severe profiles are only 12% of the default batch. They differ from
    # both adversarial holdout configs in obstacle density and mechanic stack.
    WorldProfile("dense_ridges", "severe", 1, 3500, 96, 43, 1.00, .73,
                 2750, 3500, 20, 25, 55, .72, 1.15, full_difficulty=True),
    WorldProfile("storm_crust", "severe", 1, 4000, 76, 34, .87, .66,
                 2600, 4000, 18, 28, 60, .60, 1.10,
                 stack=("Wind", "Ice", "Crust"), full_difficulty=True),
    WorldProfile("deep_water", "severe", 1, 4000, 62, 28, .81, .60,
                 2300, 2800, 17, 28, 65, .48, 1.05,
                 stack=("Liquid",), full_difficulty=True, fixed_biome_id=7),
    WorldProfile("sucking_bog", "severe", 1, 4000, 56, 24, .76, .57,
                 2200, 2500, 16, 28, 65, .46, 1.05,
                 stack=("Mud",), full_difficulty=True, fixed_biome_id=36),
    WorldProfile("thermal_marsh", "severe", 1, 4000, 52, 22, .74, .55,
                 2100, 2500, 16, 28, 65, .44, 1.05,
                 stack=("Liquid",), full_difficulty=True, fixed_biome_id=39,
                 initial_temperature=50),
)


def training_config(profile: WorldProfile, base_path: str | None = None):
    cfg = load_env_config(base_path or None)
    # Reward shaping changes only PPO's training feedback, not its observation
    # or control interface. Keep this identical across all profile cohorts.
    cfg.reward.energy_cost_scale = .025
    cfg.reward.flip_penalty = 8.0
    cfg.reward.hard_contact_penalty = 0.0
    cfg.reward.stuck_penalty = .08
    cfg.biome_split = 0
    if profile.terrain_m:
        cfg.terrain.sample_count = int(profile.terrain_m / cfg.terrain.dx) + 1
        cfg.terrain.crater_count = profile.craters
        cfg.terrain.step_count = profile.steps
        cfg.terrain.amplitude = profile.amplitude
        cfg.terrain.roughness = profile.roughness
        cfg.terrain.difficulty_distance_offset = profile.terrain_offset_m
        cfg.terrain.preserve_spawn_safety = True
        cfg.chain_zone_count = profile.biome_zones
        cfg.chain_segment_min_length = profile.zone_min_m
        cfg.chain_segment_max_length = profile.zone_max_m
        cfg.terrain_surprise_probability = profile.surprise
        cfg.terrain_surprise_strength = profile.surprise_strength
        cfg.force_full_difficulty = profile.full_difficulty
        # Biome difficulty has a separate distance scale from terrain. It is
        # possible to get hard rocks but gentle weather, and vice versa.
        # The C++ world layout uses terrain.length for this scale; save the
        # physical terrain length in sample_count/dx and set the biome ramp.
        cfg.terrain.length = profile.biome_ramp_m
        cfg.physics.initial_energy = profile.initial_energy
        cfg.physics.initial_engine_temperature = profile.initial_temperature
        cfg.fixed_biome_id = profile.fixed_biome_id
        if profile.stack:
            layers = [getattr(MechanicType, name) for name in profile.stack]
            cfg.evaluation_stack_types = layers + [MechanicType.Normal] * (4 - len(layers))
            cfg.evaluation_stack_count = len(layers)
    return cfg


def _distribute(count: int, profiles: tuple[WorldProfile, ...]) -> dict[str, int]:
    if not count:
        return {}
    total_weight = sum(profile.weight for profile in profiles)
    raw = [count * profile.weight / total_weight for profile in profiles]
    allocated = [int(value) for value in raw]
    remainder = count - sum(allocated)
    order = sorted(range(len(profiles)), key=lambda i: (raw[i] - allocated[i],
                                                         profiles[i].weight), reverse=True)
    for i in order[:remainder]:
        allocated[i] += 1
    return {profile.name: n for profile, n in zip(profiles, allocated) if n}


def allocate_profiles(num_envs: int, stress_fraction: float = .5,
                      severe_fraction: float | None = None) -> list[tuple[WorldProfile, int]]:
    if severe_fraction is None:
        severe_fraction = min(.12, stress_fraction)
    if num_envs < 1 or not 0 <= severe_fraction <= stress_fraction < 1:
        raise ValueError("require num_envs > 0 and 0 <= severe_fraction <= stress_fraction < 1")
    stress = min(num_envs - 1, round(num_envs * stress_fraction))
    severe = min(stress, round(num_envs * severe_fraction))
    tier_counts = {"accessible": num_envs - stress,
                   "challenging": stress - severe, "severe": severe}
    counts = {}
    for tier, count in tier_counts.items():
        group = tuple(profile for profile in PROFILES if profile.tier == tier)
        counts.update(_distribute(count, group))
    return [(profile, counts[profile.name]) for profile in PROFILES if profile.name in counts]
