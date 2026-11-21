/**
 * Tests for combat.ts — map-free: every decide() call injects deps.los.
 * Run: bun test tests/combat.test.ts
 */
import { test, expect } from "bun:test"
import { type ActorObs, type GameEvent, type ItemObs, type Observation, type SelfObs, type WeaponObs } from "../mechfight.ts"
import { decide, intercept, pickTarget, type CombatState, type Vec3 } from "../combat.ts"

const GUN = { damage: 45, cooldown: 0.24, speed: 252, range: 230, spread_deg: 0.6, mag: 12, reload: 2.4 }
const LASER = { damage: 315, charge: 0.75, min_charge: 0.15, hold_s: 2, range: 280, cooldown: 1.65 }
const LIGHTNING = { damage: 60, range: 45, cone_deg: 30, chain: 2, chain_range: 14, cooldown: 0.25, energy: 3 }
const MORTAR = { damage: 200, speed: 72, gravity: 22, splash: 160, splash_radius: 7, cooldown: 2 }

const LOS_TRUE = { los: () => true }
const state = (over: Partial<CombatState> = {}): CombatState => ({ gravity: 22, lastDamaged: null, ...over })

const gunWeapon = (): WeaponObs => ({ kind: "gun", cooldown: 0, ammo: 12, reload: 0, stats: GUN })

type ObsOpts = {
  tick?: number
  self?: Partial<SelfObs>
  weapons?: [WeaponObs | null, WeaponObs | null]
  mechs?: ActorObs[]
  crates?: ItemObs[]
  events?: GameEvent[]
}

function makeObs(o: ObsOpts = {}): Observation {
  const self: SelfObs = {
    id: 1,
    pos: [0, 0, 0],
    vel: [0, 0, 0],
    aim: [0, 0],
    hp: 1400,
    max_hp: 1400,
    energy: 100,
    max_energy: 100,
    grounded: true,
    cubes: 0,
    upgrades: [],
    weapons: o.weapons ?? [gunWeapon(), null],
    in_zone: true,
    stats: { muzzle_height: 7 },
    ...o.self,
  }
  return {
    type: "obs",
    tick: o.tick ?? 100,
    t: (o.tick ?? 100) / 20,
    alive: 16,
    self,
    mechs: o.mechs ?? [],
    projectiles: [],
    crates: o.crates ?? [],
    cubes: [],
    pickups: [],
    sounds: [],
    beams: [],
    events: o.events ?? [],
    zone: { center: [0, 0], radius: 560, next_center: [0, 0], next_radius: 380, shrink_start: 60, shrink_end: 110, dps: 20 },
    offer: null,
    drops: [],
  }
}

const mech = (id: number, pos: Vec3, hp = 1000, vel: Vec3 = [0, 0, 0]): ActorObs => ({
  id,
  kind: "mech",
  pos,
  vel,
  aim: [0, 0],
  hp,
  max_hp: 1400,
  cubes: 0,
  weapons: ["gun"],
  charging: false,
})

// 1. intercept math ---------------------------------------------------------

test("intercept: closed-form check across speeds, velocities and elevations", () => {
  const muzzle: Vec3 = [100, -50, 7]
  const cases: { pos: Vec3; vel: Vec3; speed: number }[] = [
    { pos: [250, 60, 0], vel: [0, 0, 0], speed: 252 }, // static
    { pos: [200, -50, 0], vel: [0, 20, 0], speed: 252 }, // perpendicular crossing
    { pos: [300, -50, 0], vel: [-20, 0, 0], speed: 252 }, // head-on
    { pos: [180, 110, 0], vel: [12, -18, 0], speed: 252 }, // diagonal
    { pos: [160, -50, 0], vel: [8, 0, 0], speed: 90 }, // slow projectile
    { pos: [220, 40, 18], vel: [10, 5, 1], speed: 252 }, // elevated + rising
  ]
  for (const c of cases) {
    const sol = intercept(muzzle, { pos: c.pos, vel: c.vel, centerHeight: 4.5 }, c.speed)
    expect(sol).not.toBeNull()
    const s = sol!

    const c0: Vec3 = [c.pos[0], c.pos[1], c.pos[2] + 4.5]
    // closed-form flight time T: |d + v·T| = speed·T  ->  a·T² + b·T + c = 0
    const d = [c0[0] - muzzle[0], c0[1] - muzzle[1], c0[2] - muzzle[2]]
    const a = c.vel[0] ** 2 + c.vel[1] ** 2 + c.vel[2] ** 2 - c.speed * c.speed
    const b = 2 * (d[0] * c.vel[0] + d[1] * c.vel[1] + d[2] * c.vel[2])
    const cc = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    const disc = b * b - 4 * a * cc
    expect(disc).toBeGreaterThanOrEqual(0)
    const r1 = (-b + Math.sqrt(disc)) / (2 * a)
    const r2 = (-b - Math.sqrt(disc)) / (2 * a)
    const T = Math.max(r1, r2) // roots have opposite signs; the positive one is the flight time
    expect(T).toBeGreaterThan(0)

    const pred: Vec3 = [c0[0] + c.vel[0] * T, c0[1] + c.vel[1] * T, c0[2] + c.vel[2] * T]
    // a bullet launched along the solution at `speed` is at the predicted point at time T
    const dir = [
      Math.cos(s.pitch) * Math.cos(s.yaw),
      Math.cos(s.pitch) * Math.sin(s.yaw),
      Math.sin(s.pitch),
    ]
    const bullet: Vec3 = [
      muzzle[0] + dir[0] * c.speed * T,
      muzzle[1] + dir[1] * c.speed * T,
      muzzle[2] + dir[2] * c.speed * T,
    ]
    const miss = Math.hypot(bullet[0] - pred[0], bullet[1] - pred[1], bullet[2] - pred[2])
    expect(miss).toBeLessThan(0.3)

    // the solution angles point at the predicted point within 0.5 deg
    const yaw = Math.atan2(pred[1] - muzzle[1], pred[0] - muzzle[0])
    const pitch = Math.atan2(pred[2] - muzzle[2], Math.hypot(pred[0] - muzzle[0], pred[1] - muzzle[1]))
    expect(Math.abs(s.yaw - yaw)).toBeLessThan((0.5 * Math.PI) / 180)
    expect(Math.abs(s.pitch - pitch)).toBeLessThan((0.5 * Math.PI) / 180)
  }
})

// 2. intercept null / edge ---------------------------------------------------

test("intercept: unreachable chases return null", () => {
  expect(intercept([0, 0, 7], { pos: [100, 0, 0], vel: [260, 0, 0], centerHeight: 4.5 }, 252)).toBeNull() // faster than bullet
  expect(intercept([0, 0, 7], { pos: [100, 0, 0], vel: [252, 0, 0], centerHeight: 4.5 }, 252)).toBeNull() // exactly bullet speed
  expect(intercept([0, 0, 7], { pos: [100, 0, 0], vel: [0, 0, 0], centerHeight: 4.5 }, 0)).toBeNull() // no projectile speed
})

// 3. gun fire gate -----------------------------------------------------------

test("decide gun: holds fire while off the lead yaw, fires when aligned", () => {
  const target = mech(5, [60, 10, 0])

  const off = decide(makeObs({ self: { aim: [0.5, 0] }, mechs: [target] }), state(), LOS_TRUE)
  expect(off).not.toBeNull()
  expect(off!.fire).toEqual([false, false])
  expect(off!.target).toEqual({ id: 5, kind: "mech" })
  expect(off!.aim[0]).toBeCloseTo(Math.atan2(10, 60), 5) // still aims at the target while gating

  const on = decide(makeObs({ self: { aim: [Math.atan2(10, 60), 0] }, mechs: [target] }), state(), LOS_TRUE)
  expect(on!.fire).toEqual([true, false])
  expect(on!.reload).toBe(false)
})

// 4. laser hold / release ----------------------------------------------------

test("decide laser: hold below full charge, release at full with clear line, auto-release past hold_s", () => {
  const target = mech(7, [50, 0, 0]) // yaw from self = 0
  const laser = (charge: number): WeaponObs => ({ kind: "laser", cooldown: 0, charge, stats: LASER })
  const withLaser = (w: WeaponObs): [WeaponObs | null, WeaponObs | null] => [gunWeapon(), w]

  // charging, aim already on target: keep holding both gun and laser
  const charging = decide(
    makeObs({ self: { aim: [0, 0] }, weapons: withLaser(laser(0.3)), mechs: [target] }),
    state(),
    LOS_TRUE,
  )
  expect(charging!.fire).toEqual([true, true])

  // full charge + aligned + clear los: release (fire[1] false) with the aim on target
  const full = decide(
    makeObs({ self: { aim: [0, 0] }, weapons: withLaser(laser(0.75)), mechs: [target] }),
    state(),
    LOS_TRUE,
  )
  expect(full!.fire[1]).toBe(false)
  expect(full!.aim[0]).toBeCloseTo(0, 5)

  // held at full charge for hold_s = 2 s = 40 ticks with a blocked release: let go anyway
  const held = state({ laserFullTick: 60 })
  const stuck = decide(
    makeObs({ tick: 100, self: { aim: [2.0, 0] }, weapons: withLaser(laser(0.75)), mechs: [target] }),
    held,
    LOS_TRUE,
  )
  expect(stuck!.fire[1]).toBe(false)
})

// 5. lightning ----------------------------------------------------------------

test("decide lightning: in range + cone with energy fires; low energy does not", () => {
  const target = mech(3, [40, 0, 0]) // 40 m < 0.95 * 45, dead ahead of aim yaw 0
  const withLightning = (): [WeaponObs | null, WeaponObs | null] => [
    gunWeapon(),
    { kind: "lightning", cooldown: 0, stats: LIGHTNING },
  ]

  const hi = decide(makeObs({ self: { aim: [0, 0], energy: 50 }, weapons: withLightning(), mechs: [target] }), state(), LOS_TRUE)
  expect(hi!.fire[1]).toBe(true)

  const lo = decide(makeObs({ self: { aim: [0, 0], energy: 10 }, weapons: withLightning(), mechs: [target] }), state(), LOS_TRUE)
  expect(lo!.fire[1]).toBe(false)

  // target behind a wall: don't spend energy arcing into it
  const walled = decide(
    makeObs({ self: { aim: [0, 0], energy: 50 }, weapons: withLightning(), mechs: [target] }),
    state(),
    { los: () => false },
  )
  expect(walled!.fire[1]).toBe(false)
})

// 6. mortar gating --------------------------------------------------------------

const withMortar = (): [WeaponObs | null, WeaponObs | null] => [gunWeapon(), { kind: "mortar", cooldown: 0, stats: MORTAR }]

test("decide mortar: no shell under 40 m; at range torso owns the lob, fires only once the actual pitch is on it", () => {
  const close = decide(makeObs({ self: { aim: [0, 0] }, weapons: withMortar(), mechs: [mech(9, [30, 0, 0])] }), state(), LOS_TRUE)
  expect(close!.fire[1]).toBe(false)

  // torso still flat: command the lob but don't waste a shell
  const d = decide(makeObs({ self: { aim: [0, 0] }, weapons: withMortar(), mechs: [mech(9, [100, 0, 0])] }), state(), LOS_TRUE)!
  expect(d.fire[1]).toBe(false)
  // low-arc ballistic solution: R = 100, tube z = 9.5, center z = 4.5
  const k = (72 * 72) / (22 * 100)
  const disc = k * k - 1 - (2 * 72 * 72 * (4.5 - 9.5)) / (22 * 100 * 100)
  expect(d.aim[1]).toBeCloseTo(Math.atan(k - Math.sqrt(disc)), 3)
  expect(d.aim[1]).toBeGreaterThan(0) // lob pitch, not the gun's slightly negative pitch
  expect(d.aim[0]).toBeCloseTo(0, 3)

  // torso settled on the lob: shell away
  const settled = decide(makeObs({ self: { aim: d.aim }, weapons: withMortar(), mechs: [mech(9, [100, 0, 0])] }), state(), LOS_TRUE)!
  expect(settled.fire[1]).toBe(true)
})

// regression (review I1): gun + mortar at 100 m — the gun must not fire while the mortar owns the torso
test("regression: gun holds fire while the mortar owns the torso at 100 m", () => {
  const target = mech(9, [100, 0, 0])
  const first = decide(makeObs({ self: { aim: [0, 0] }, weapons: withMortar(), mechs: [target] }), state(), LOS_TRUE)!
  expect(first.fire[0]).toBe(false) // gun yaw is aligned, but the torso is being driven to the lob
  const onLob = decide(makeObs({ self: { aim: first.aim }, weapons: withMortar(), mechs: [target] }), state(), LOS_TRUE)!
  expect(onLob.fire).toEqual([false, true]) // bullets along the lob arc would miss high
})

// regression (review C1): laser vs crossing target at 150 m — release on the CURRENT position, not the gun lead
test("regression: laser releases on a crossing 150 m target once the torso settles on the commanded aim", () => {
  const vel: Vec3 = [0, 11.7, 0] // walk speed, perpendicular
  const target = mech(7, [150, 0, 0], 1000, vel)
  const weapons: [WeaponObs | null, WeaponObs | null] = [gunWeapon(), { kind: "laser", cooldown: 0, charge: 0.75, stats: LASER }]

  const first = decide(makeObs({ self: { aim: [0.5, 0] }, weapons, mechs: [target] }), state(), LOS_TRUE)!
  // commanded aim is the direct (unled) yaw, which is outside the laser gate from the gun lead yaw
  expect(first.aim[0]).toBeCloseTo(0, 5)
  const lead = intercept([0, 0, 7], { pos: target.pos, vel, centerHeight: 4.5 }, GUN.speed)!
  expect(Math.abs(lead.yaw - first.aim[0])).toBeGreaterThan(Math.atan(2.5 / 150))
  expect(first.fire[1]).toBe(true) // not on target yet: keep holding

  const settled = decide(makeObs({ self: { aim: first.aim }, weapons, mechs: [target] }), state(), LOS_TRUE)!
  expect(settled.fire[1]).toBe(false) // released at the target's current position
})

// 7. rear-damage response -------------------------------------------------------

test("decide: keeps turning to unseen damage across ticks for the 2 s (40 tick) window, then clears it", () => {
  const ev: GameEvent = { type: "damaged", amount: 90, weapon: "gun", bearing: 1.2, by: 3 }
  const s = state()
  const d = decide(makeObs({ tick: 500, events: [ev] }), s, LOS_TRUE)!
  expect(d.aim).toEqual([1.2, 0])
  expect(d.fire).toEqual([false, false])
  expect(d.target).toBeNull()
  expect(s.lastDamaged).toEqual({ bearing: 1.2, byId: 3, tObsTick: 500 }) // not cleared on first use

  // next decisions without events: keep slewing to the bearing, not freezing on the mid-turn aim
  const d2 = decide(makeObs({ tick: 502, self: { aim: [0.4, 0] } }), s, LOS_TRUE)!
  expect(d2.aim).toEqual([1.2, 0])

  // exactly 40 ticks old: still fresh
  const d3 = decide(makeObs({ tick: 540, self: { aim: [1.0, 0] } }), s, LOS_TRUE)!
  expect(d3.aim).toEqual([1.2, 0])

  // 41 ticks old: expired — coast on the current aim and drop the memory
  const d4 = decide(makeObs({ tick: 541, self: { aim: [0.3, -0.1] } }), s, LOS_TRUE)!
  expect(d4.aim).toEqual([0.3, -0.1])
  expect(d4.fire).toEqual([false, false])
  expect(s.lastDamaged).toBeNull()
})

// pickTarget priority ------------------------------------------------------------

test("pickTarget: mechs first (attacker > lowest hp > nearest), then NPC, then crate", () => {
  const npc = (id: number, pos: Vec3, hp = 350): ActorObs => ({ id, kind: "npc", pos, vel: [0, 0, 0], aim: [0, 0], hp, max_hp: 350 })

  const two = makeObs({ mechs: [mech(2, [30, 0, 0], 500), mech(8, [80, 0, 0], 1400)] })
  expect(pickTarget(two, state({ lastDamaged: { bearing: 0, tObsTick: 100, byId: 8 } }))?.id).toBe(8) // attacker wins
  expect(pickTarget(two, state())?.id).toBe(2) // lowest hp
  const tied = makeObs({ mechs: [mech(2, [50, 0, 0], 500), mech(8, [30, 0, 0], 500)] })
  expect(pickTarget(tied, state())?.id).toBe(8) // nearest on hp tie

  expect(pickTarget(makeObs({ mechs: [npc(104, [10, 5, 0])] }), state())?.kind).toBe("npc")
  const crates = makeObs({ crates: [{ id: 1007, pos: [12, -40, 0], hp: 300 }] })
  expect(pickTarget(crates, state())).toEqual({ id: 1007, kind: "crate", pos: [12, -40, 0], hp: 300 })
  expect(pickTarget(makeObs(), state())).toBeNull()
})

// reload when nothing is visible -------------------------------------------------

test("decide: reloads the gun when no target is visible and ammo is low", () => {
  const low = decide(
    makeObs({ self: { aim: [0.4, 0.1] }, weapons: [{ kind: "gun", cooldown: 0, ammo: 2, reload: 0, stats: GUN }, null] }),
    state(),
    LOS_TRUE,
  )
  expect(low!.reload).toBe(true)
  expect(low!.aim).toEqual([0.4, 0.1]) // aim unchanged
  expect(low!.fire).toEqual([false, false])
  expect(low!.target).toBeNull()

  const full = decide(makeObs({ self: { aim: [0.4, 0.1] } }), state(), LOS_TRUE)
  expect(full!.reload).toBe(false)
})
