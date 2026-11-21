/**
 * sim/config.ts — loads the complete rule set from the recorded hello (first line of
 * kit/check/stream.jsonl.gz). The engine reads every number from here; nothing hard-coded.
 */
import { readFileSync } from "node:fs"
import { gunzipSync } from "node:zlib"
import { join } from "node:path"

export type Cfg = {
  match: { tick_hz: number; decide_every: number; duration_s: number; mechs: number; spawn_min_separation: number; spawn_zone_fraction: number; arena_margin: number }
  bot: { tick_budget_ms: number; hello_budget_ms: number; coast_s: number; offer_timeout_s: number }
  mech: {
    radius: number; height: number; eye_height: number; muzzle_height: number; hp: number; walk_speed: number
    boost_speed: number; ground_accel: number; air_accel: number; gravity: number; jump_speed: number
    jet_accel: number; max_rise_speed: number; step_up: number; energy: number; energy_regen: number
    energy_regen_delay: number; boost_drain: number; jet_drain: number; turn_rate: number; pitch_rate: number
    regen_delay: number; regen_per_s: number; rear_mult: number; front_arc_deg: number; self_damage_mult: number
    hearing: number; min_pitch: number; max_pitch: number; max_slope_deg: number
    [k: string]: number
  }
  vision: { range: number; fov_deg: number; near_range: number; show_enemy_hp: boolean; memory_s: number }
  sound: { bearing_noise_deg: number; near: number; mid: number; step_speed: number; step_every: number; range: Record<string, number> }
  weapons: Record<string, Record<string, number>>
  cubes: { pickup_radius: number; hp_per_cube: number; damage_per_cube: number; stat_cap: number; size_per_cube: number; upgrade_thresholds: number[] }
  crates: { count: number; roof_fraction: number; hp: number; radius: number; height: number; cubes: number }
  loot: { pickup_radius: number; equip_s: number; floor_count: number; floor_roof_fraction: number; crate_weapon_chance: number; weights: Record<string, number> }
  npc: { count: number; hp: number; radius: number; height: number; speed: number; leash: number; chase_leash: number; aggro_range: number; deaggro_range: number; damage: number; cooldown: number; projectile_speed: number; spread_deg: number; cubes: number }
  drops: { times: number[]; fall_s: number; radius: number; capture_s: number; cubes: number; spread: number }
  zone: { start_center: [number, number]; start_radius: number; phases: { hold_s: number; shrink_s: number; radius: number; dps: number }[] }
  upgrades: { id: string; effects: { path: string; mul?: number; add?: number }[] }[]
}
export type Hello = { type: "hello"; protocol: string; slot: number; seed: number; map: { dir: string; files: string[] }; config: Cfg }

const KIT_DIR = join(import.meta.dir, "../../kit")

export function loadHello(streamPath?: string): Hello {
  const p = streamPath ?? join(KIT_DIR, "check/stream.jsonl.gz")
  const first = gunzipSync(readFileSync(p)).toString("utf8").split("\n")[0]!
  const hello = JSON.parse(first) as Hello
  if (hello.type !== "hello") throw new Error(`not a hello: ${hello.type}`)
  return hello
}

export const arenaDir = (): string => process.env.STRESS_ARENA ?? join(KIT_DIR, "arena")

/** mulberry32; deterministic given the match seed. */
export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** standard normal, Box-Muller */
export function gauss(r: () => number): number {
  const u = Math.max(r(), 1e-12)
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r())
}

export const angDiff = (a: number, b: number): number => {
  let d = a - b
  while (d > Math.PI) d -= 2 * Math.PI
  while (d < -Math.PI) d += 2 * Math.PI
  return d
}
