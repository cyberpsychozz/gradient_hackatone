"""Smoke checks for the 20 appended training worlds.

Run after rebuilding the editable native extension:
    python -m unittest discover -s tests -p test_custom_worlds.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

from _mars_rover_cpp import MarsRoverBatchEnv, biome_catalog
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import load_env_config


PUBLIC_IDS = (
    "normal", "sand", "ice", "mud", "wind", "low_gravity", "crust", "liquid",
    "collapse_window_flats", "collapse_window_gulch", "setpoint_rime_shelf",
    "collapse_window_playa", "pulse_gravity_reef", "commitment_ledge_field",
    "lateral_shear_belt", "hysteresis_surge_bog", "gravity_shelf_lug",
    "thermal_surge_relay", "rime_quarry_dawn", "gravity_shear_escarpment",
)

CUSTOM_IDS = (
    "sunlit_gravel_plain", "amber_basalt_track", "pale_dune_lane",
    "morning_breeze_flat", "shallow_frost_plain", "wide_mesa_approach",
    "rolling_iron_hills", "ochre_switchbacks", "crosswind_causeway",
    "cold_mirror_ridge", "soft_silt_fan", "broken_shale_steps",
    "low_g_craterfield", "dusk_sand_ripples", "basalt_teeth_gorge",
    "stormglass_scree", "sucking_clay_basin", "magnetic_gust_corridor",
    "broken_ledge_plateau", "eclipse_thermal_marsh",
)


def fixed_world(index: int) -> MarsRoverBatchEnv:
    config = load_env_config()
    config.fixed_biome_id = index
    config.chain_biomes = False
    config.terrain.sample_count = 1600  # 400 m, fast enough for repeated resets.
    config.difficulty_safe_fraction_min = 0.0
    config.difficulty_safe_fraction_max = 0.0
    return MarsRoverBatchEnv(1, config)


class CustomWorldsTest(unittest.TestCase):
    def test_catalog_and_contract(self) -> None:
        catalog = biome_catalog()
        self.assertEqual(len(catalog), 40)
        self.assertEqual(tuple(world["id"] for world in catalog[:20]), PUBLIC_IDS)
        self.assertEqual(tuple(world["id"] for world in catalog[20:]), CUSTOM_IDS)
        self.assertEqual(len({world["id"] for world in catalog}), 40)
        self.assertEqual(
            [sum(world["skill_stratum"] in grades for world in catalog[20:])
             for grades in (("G1", "G2"), ("G3",), ("G4", "G5"))],
            [6, 8, 6],
        )
        self.assertTrue(all(world["split"] == 1 for world in catalog[20:]))
        self.assertEqual(len(ACTION_MACROS), 31)
        # Vary at least the main conditions, not just the labels or colours.
        signatures = {
            tuple(round(float(world["parameters"][key]), 3) for key in
                  ("friction", "sink", "viscosity", "wind", "gravity",
                   "temperature", "solar", "lidar_range"))
            for world in catalog[20:]
        }
        self.assertEqual(len(signatures), 20)

    def test_every_world_resets_and_steps(self) -> None:
        catalog = biome_catalog()
        observations = np.empty((1, 160), dtype=np.float32)
        rewards = np.empty(1, dtype=np.float32)
        terminated = np.empty(1, dtype=np.uint8)
        truncated = np.empty(1, dtype=np.uint8)
        actions = np.empty(1, dtype=np.int32)
        for index in range(20, 40):
            with self.subTest(world=catalog[index]["id"]):
                env = fixed_world(index)
                self.assertEqual(env.obs_dim, 160)
                for seed in (42, 2026):
                    env.reset_at(0, seed, True, observations[0])
                    info = env.debug_info(0)
                    self.assertEqual(info["mechanic"], catalog[index]["name"])
                    self.assertTrue(np.isfinite(observations).all())
                    for macro_index in (1, 1, 1, 12, 0, 1, 1, 1):
                        actions[0] = ACTION_MACROS[macro_index]
                        env.step(actions, observations, rewards, terminated, truncated)
                        self.assertTrue(np.isfinite(observations).all())
                        self.assertTrue(np.isfinite(rewards).all())
                        if terminated[0] or truncated[0]:
                            break
                    info = env.debug_info(0)
                    self.assertTrue(np.isfinite((info["x"], info["energy"])).all())

    def test_sticky_world_changes_phase(self) -> None:
        env = fixed_world(36)
        observations = np.empty((1, 160), dtype=np.float32)
        rewards = np.empty(1, dtype=np.float32)
        terminated = np.empty(1, dtype=np.uint8)
        truncated = np.empty(1, dtype=np.uint8)
        actions = np.asarray([ACTION_MACROS[0]], dtype=np.int32)
        env.reset_at(0, 42, True, observations[0])
        self.assertEqual(env.debug_info(0)["hazard"], 1)
        for _ in range(121):
            env.step(actions, observations, rewards, terminated, truncated)
        self.assertEqual(env.debug_info(0)["hazard"], 0)
        self.assertTrue(np.isfinite(observations).all())

    def test_ledge_world_adds_obstacles(self) -> None:
        obs = np.empty(160, dtype=np.float32)
        plain = fixed_world(20)
        ledges = fixed_world(38)
        plain.reset_at(0, 42, True, obs)
        plain_pits = plain.debug_info(0)["generated_pit_count"]
        ledges.reset_at(0, 42, True, obs)
        self.assertGreater(ledges.debug_info(0)["generated_pit_count"], plain_pits)
        self.assertGreater(biome_catalog()[38]["parameters"]["ledge_gap_width"], 0)


if __name__ == "__main__":
    unittest.main()
