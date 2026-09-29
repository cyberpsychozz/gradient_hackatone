#pragma once

// Extra training worlds. Appended after the public bank so existing biome IDs stay stable.
namespace custom_biomes {

enum class WorldEffect { None, Gust, Sticky, PulseDrag };

struct WorldSpec {
  std::string_view id;
  std::string_view name;
  std::string_view grade;
  MechanicType type;
  float grip, sink, viscosity, wind, gravity, energy;
  float temperature, thermal, solar, lidar_cost, lidar_range;
  float amplitude, roughness, craters, steps, deform;
  float wave_height, wave_period, ledge_width;
  WorldEffect effect;
  float effect_strength;
};

using T = MechanicType;
using E = WorldEffect;
// ID, name, grade, type | soil/weather/energy/sensors | terrain | timed effect.
inline constexpr std::array<WorldSpec, 20> kWorlds{{
    {"sunlit_gravel_plain", "Sunlit Gravel Plain", "G1", T::Normal,
     1.08f, 0.0f, 0.0f, 0.0f, 1.0f, 0.90f, -8.0f, 0.90f, 1.55f, 0.80f, 1.00f,
     0.60f, 0.65f, 0.55f, 0.55f, 0.0f, 0.03f, 18.0f, 0.0f, E::None, 0.0f},
    {"amber_basalt_track", "Amber Basalt Track", "G1", T::Crust,
     1.18f, 0.0f, 0.0f, 0.0f, 1.0f, 0.95f, -20.0f, 1.0f, 1.35f, 0.85f, 1.0f,
     0.70f, 0.70f, 0.60f, 0.55f, 0.003f, 0.05f, 15.0f, 0.0f, E::None, 0.0f},
    {"pale_dune_lane", "Pale Dune Lane", "G2", T::Sand,
     0.82f, 0.003f, 0.30f, 0.0f, 1.0f, 1.10f, 4.0f, 0.95f, 1.30f, 1.0f, 1.0f,
     0.75f, 0.80f, 0.65f, 0.65f, 0.0f, 0.10f, 11.0f, 0.0f, E::None, 0.0f},
    {"morning_breeze_flat", "Morning Breeze Flat", "G2", T::Wind,
     1.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f, -12.0f, 1.0f, 1.20f, 1.0f, 1.0f,
     0.75f, 0.75f, 0.70f, 0.65f, 0.0f, 0.05f, 17.0f, 0.0f, E::None, 0.0f},
    {"shallow_frost_plain", "Shallow Frost Plain", "G2", T::Ice,
     0.72f, 0.0f, 0.0f, 0.0f, 1.0f, 1.05f, -35.0f, 1.20f, 1.40f, 1.0f, 1.0f,
     0.78f, 0.80f, 0.70f, 0.70f, 0.0f, 0.08f, 14.0f, 0.0f, E::None, 0.0f},
    {"wide_mesa_approach", "Wide Mesa Approach", "G2", T::LowGravity,
     1.10f, 0.0f, 0.0f, 0.0f, 0.82f, 1.0f, -24.0f, 0.90f, 1.30f, 1.0f, 1.0f,
     0.70f, 0.75f, 0.70f, 0.65f, 0.0f, 0.08f, 20.0f, 0.0f, E::None, 0.0f},
    {"rolling_iron_hills", "Rolling Iron Hills", "G3", T::Normal,
     1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.10f, -18.0f, 1.0f, 1.0f, 1.0f, 0.9f,
     1.30f, 1.40f, 1.20f, 1.30f, 0.0f, 0.35f, 12.0f, 0.0f, E::None, 0.0f},
    {"ochre_switchbacks", "Ochre Switchbacks", "G3", T::Crust,
     0.92f, 0.004f, 0.20f, 0.0f, 1.0f, 1.12f, 16.0f, 1.10f, 1.10f, 1.0f, 0.85f,
     1.20f, 1.25f, 1.0f, 1.70f, 0.012f, 0.20f, 9.0f, 0.0f, E::None, 0.0f},
    {"crosswind_causeway", "Crosswind Causeway", "G3", T::Wind,
     0.93f, 0.0f, 0.0f, 8.0f, 1.0f, 1.18f, -27.0f, 1.30f, 0.90f, 1.15f, 0.80f,
     0.85f, 1.0f, 0.90f, 0.90f, 0.0f, 0.12f, 16.0f, 0.0f, E::Gust, 0.08f},
    {"cold_mirror_ridge", "Cold Mirror Ridge", "G3", T::Ice,
     0.44f, 0.0f, 0.0f, 0.0f, 1.0f, 1.20f, -50.0f, 2.0f, 1.10f, 1.25f, 0.75f,
     1.10f, 1.30f, 1.15f, 1.10f, 0.0f, 0.15f, 12.0f, 0.0f, E::None, 0.0f},
    {"soft_silt_fan", "Soft Silt Fan", "G3", T::Mud,
     0.65f, 0.018f, 1.10f, 0.0f, 1.0f, 1.35f, 2.0f, 1.15f, 0.80f, 1.15f, 0.85f,
     0.95f, 1.0f, 0.80f, 0.80f, 0.0f, 0.12f, 14.0f, 0.0f, E::None, 0.0f},
    {"broken_shale_steps", "Broken Shale Steps", "G3", T::Crust,
     0.90f, 0.0f, 0.15f, 0.0f, 1.0f, 1.20f, -14.0f, 1.0f, 0.95f, 1.0f, 0.80f,
     1.15f, 1.60f, 1.10f, 2.0f, 0.015f, 0.24f, 7.0f, 0.0f, E::None, 0.0f},
    {"low_g_craterfield", "Low G Craterfield", "G3", T::LowGravity,
     0.90f, 0.0f, 0.0f, 0.0f, 0.58f, 1.10f, -32.0f, 1.0f, 1.10f, 1.0f, 0.90f,
     1.15f, 1.30f, 2.20f, 1.10f, 0.0f, 0.18f, 18.0f, 0.0f, E::None, 0.0f},
    {"dusk_sand_ripples", "Dusk Sand Ripples", "G3", T::Sand,
     0.58f, 0.016f, 0.75f, 3.0f, 1.0f, 1.45f, -9.0f, 1.10f, 0.55f, 1.20f, 0.80f,
     1.10f, 1.35f, 0.90f, 0.90f, 0.0f, 0.20f, 4.0f, 0.0f, E::None, 0.0f},
    {"basalt_teeth_gorge", "Basalt Teeth Gorge", "G4", T::Normal,
     0.85f, 0.0f, 0.0f, 0.0f, 1.10f, 1.35f, -31.0f, 1.20f, 0.75f, 1.60f, 0.65f,
     1.45f, 1.70f, 1.80f, 2.50f, 0.0f, 0.48f, 5.0f, 0.0f, E::None, 0.0f},
    {"stormglass_scree", "Stormglass Scree", "G4", T::Ice,
     0.30f, 0.0f, 0.0f, 9.0f, 0.95f, 1.50f, -58.0f, 2.40f, 0.45f, 1.70f, 0.60f,
     1.25f, 1.50f, 1.20f, 1.60f, 0.0f, 0.28f, 8.0f, 0.0f, E::None, 0.0f},
    {"sucking_clay_basin", "Sucking Clay Basin", "G4", T::Mud,
     0.45f, 0.035f, 2.10f, 0.0f, 1.05f, 1.75f, 7.0f, 1.45f, 0.50f, 1.80f, 0.65f,
     1.15f, 1.30f, 0.75f, 0.70f, 0.0f, 0.12f, 12.0f, 0.0f, E::Sticky, 0.35f},
    {"magnetic_gust_corridor", "Magnetic Gust Corridor", "G4", T::Wind,
     0.78f, 0.0f, 0.0f, 12.0f, 1.10f, 1.50f, -39.0f, 1.60f, 0.35f, 2.0f, 0.45f,
     1.10f, 1.15f, 1.20f, 1.10f, 0.0f, 0.18f, 10.0f, 0.0f, E::Gust, 0.30f},
    {"broken_ledge_plateau", "Broken Ledge Plateau", "G5", T::Crust,
     1.05f, 0.0f, 0.0f, 0.0f, 0.85f, 1.35f, -23.0f, 1.0f, 0.60f, 1.55f, 0.75f,
     0.90f, 1.15f, 0.75f, 0.70f, 0.005f, 0.10f, 16.0f, 2.40f, E::None, 0.0f},
    {"eclipse_thermal_marsh", "Eclipse Thermal Marsh", "G5", T::Liquid,
     0.50f, 0.020f, 1.20f, 0.0f, 0.95f, 1.80f, 50.0f, 2.80f, 0.14f, 2.20f, 0.40f,
     1.10f, 1.30f, 0.80f, 0.80f, 0.0f, 0.13f, 13.0f, 0.0f, E::PulseDrag, 0.45f},
}};

class TrainingWorld final : public Biome {
 public:
  explicit TrainingWorld(const WorldSpec& spec) : spec_(spec) {}
  std::string_view id() const noexcept override { return spec_.id; }
  std::string_view display_name() const noexcept override { return spec_.name; }
  std::string_view skill_stratum() const noexcept override { return spec_.grade; }
  MechanicType visual_type() const noexcept override { return spec_.type; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }

  MechanicParams sample_params(uint64_t seed) const noexcept override {
    MechanicParams p;
    const float jitter = biome_random01(seed, 1) - 0.5f;
    p.friction_mul = spec_.grip * (1.0f + 0.06f * jitter);
    p.sink_rate = spec_.sink;
    p.viscosity = spec_.viscosity;
    p.wind_force = spec_.wind * (1.0f + 0.10f * jitter);
    p.gravity_mul = spec_.gravity;
    p.energy_drain_mul = spec_.energy;
    p.ambient_temperature = spec_.temperature + 4.0f * jitter;
    p.thermal_transfer = spec_.thermal;
    p.solar_charge_rate = spec_.solar;
    p.lidar_energy_mul = spec_.lidar_cost;
    p.lidar_range_mul = spec_.lidar_range;
    p.terrain_amplitude_mul = spec_.amplitude;
    p.terrain_roughness_mul = spec_.roughness;
    p.terrain_crater_mul = spec_.craters;
    p.terrain_step_mul = spec_.steps;
    p.crust_deform = spec_.deform;
    if (spec_.ledge_width > 0.0f) {
      p.ledge_gap_width = spec_.ledge_width;
      p.ledge_spacing = 34.0f;
      p.ledge_ramp_length = 3.0f;
      p.ledge_ramp_height = 0.60f;
      p.ledge_start_x = 22.0f;
    }
    return p;
  }

  float terrain_height_delta(float x, uint64_t seed) const noexcept override {
    if (spec_.wave_height <= 0.0f) return 0.0f;
    const float phase = 6.2831853f * biome_random01(seed, 7);
    const float angle = 6.2831853f * std::fmod(x, spec_.wave_period) / spec_.wave_period + phase;
    return spec_.wave_height * (0.75f * std::sin(angle) + 0.25f * std::sin(2.0f * angle));
  }

  int hazard_at(int step) const noexcept override {
    return spec_.effect == E::Gust ||
                   (spec_.effect != E::None && active(step))
               ? 1 : 0;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (spec_.effect != E::Sticky || !active(c.step_index) ||
        !c.contact || !c.contact->active) return;
    c.contact->penetration += spec_.effect_strength * 0.12f * c.dt;
    if (c.wheel_force)
      *c.wheel_force += c.contact->tangent * (-5.0f * spec_.effect_strength * c.wheel_speed);
    if (c.energy_cost)
      *c.energy_cost += 0.10f * spec_.effect_strength * p.energy_drain_mul * c.dt;
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!c.body_force) return;
    if (spec_.effect == E::Gust) {
      const float pulse = std::sin(0.035f * static_cast<float>(c.step_index));
      c.body_force->x += spec_.effect_strength * c.mass * std::abs(c.gravity) * pulse;
      if (c.body_torque)
        *c.body_torque += 0.18f * spec_.effect_strength * c.mass * std::abs(c.gravity) * pulse;
    } else if (spec_.effect == E::PulseDrag && active(c.step_index)) {
      c.body_force->x -= spec_.effect_strength * c.mass * c.velocity.x * 0.25f;
      if (c.body_torque)
        *c.body_torque -= spec_.effect_strength * c.mass * c.angular_velocity * 0.08f;
    }
  }

 private:
  bool active(int step) const noexcept { return (step % 360) < 120; }
  const WorldSpec& spec_;
};

inline bool is_custom_biome_id(int id) noexcept {
  return id >= 20 && id < 20 + static_cast<int>(kWorlds.size());
}

inline void append(std::vector<const Biome*>& out) {
  static const std::array<TrainingWorld, kWorlds.size()> worlds{{
      TrainingWorld(kWorlds[0]), TrainingWorld(kWorlds[1]), TrainingWorld(kWorlds[2]),
      TrainingWorld(kWorlds[3]), TrainingWorld(kWorlds[4]), TrainingWorld(kWorlds[5]),
      TrainingWorld(kWorlds[6]), TrainingWorld(kWorlds[7]), TrainingWorld(kWorlds[8]),
      TrainingWorld(kWorlds[9]), TrainingWorld(kWorlds[10]), TrainingWorld(kWorlds[11]),
      TrainingWorld(kWorlds[12]), TrainingWorld(kWorlds[13]), TrainingWorld(kWorlds[14]),
      TrainingWorld(kWorlds[15]), TrainingWorld(kWorlds[16]), TrainingWorld(kWorlds[17]),
      TrainingWorld(kWorlds[18]), TrainingWorld(kWorlds[19]),
  }};
  for (const TrainingWorld& world : worlds) out.push_back(&world);
}

}  // namespace custom_biomes
