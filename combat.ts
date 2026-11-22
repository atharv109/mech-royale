/**
 * Combat module (Task 3): target selection, lead solving, and per-tick fire decisions.
 *
 * Conventions (kit protocol §5): x east, y north, z up; yaw 0 = +x counter-clockwise,
 * pitch up+. A direction is [cos(p)·cos(y), cos(p)·sin(y), sin(p)]. Mech pos is the feet.
 *
 * All weapon numbers come from the live stats in the observation, never hard-coded.
 */
import { type Observation } from "./mechfight.ts"

export type Vec3 = [number, number, number]

export interface CombatState {
  /** mortar gravity; brain sets it from hello.config.weapons.mortar.gravity (default 22) */
  gravity: number
  /** set on 'damaged' events with a bearing; cleared once older than the 2 s (40 tick) window */
  lastDamaged: { bearing: number; tObsTick: number; byId: number | null } | null
  /** internal: tick when the laser was first seen at full charge (hold_s auto-release) */
  laserFullTick?: number | null
  /** cb/trigger: obs.tick until which the gun/lightning trigger must stay held */
  gunHot?: number | null
  lightHot?: number | null
}

export interface CombatDecision {
  aim: [number, number]
  fire: [boolean, boolean]
  reload: boolean
  target: { id: number; kind: "mech" | "npc" | "crate" } | null
}

const MECH_RADIUS = 2.5
const MECH_CENTER = 4.5
const SHOT_OUT = 3.0 // every shot leaves radius + 0.5 = 3.0 m out along the aim (protocol §5)
const TICKS_PER_S = 20
const DAMAGE_WINDOW_TICKS = 2 * TICKS_PER_S // unseen-attacker bearing memory: 2 s
const MORTAR_PITCH_TOL = 0.05 // rad: actual torso pitch must be this close to the lob before firing

const normAngle = (a: number): number => {
  a %= 2 * Math.PI
  if (a > Math.PI) a -= 2 * Math.PI
  else if (a < -Math.PI) a += 2 * Math.PI
  return a
}

const dirOf = (yaw: number, pitch: number): Vec3 => [
  Math.cos(pitch) * Math.cos(yaw),
  Math.cos(pitch) * Math.sin(yaw),
  Math.sin(pitch),
]

const dist3 = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
const horizDist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1])

export function pickTarget(
  obs: Observation,
  state: CombatState,
): { id: number; kind: "mech" | "npc" | "crate"; pos: Vec3; hp: number } | null {
  const self = obs.self.pos
  // lower hp wins; on an hp tie (or missing hp) the nearer one wins
  const better = (best: { pos: Vec3; hp?: number }, m: { pos: Vec3; hp?: number }) => {
    const bh = best.hp ?? Infinity
    const mh = m.hp ?? Infinity
    return mh < bh || (mh === bh && horizDist(m.pos, self) < horizDist(best.pos, self))
  }

  const mechs = obs.mechs.filter((m) => m.kind === "mech")
  if (mechs.length > 0) {
    const byId = state.lastDamaged?.byId
    if (byId != null) {
      const attacker = mechs.find((m) => m.id === byId)
      if (attacker) return { id: attacker.id, kind: "mech", pos: attacker.pos, hp: attacker.hp ?? 0 }
    }
    let best = mechs[0]
    for (const m of mechs) if (better(best, m)) best = m
    return { id: best.id, kind: "mech", pos: best.pos, hp: best.hp ?? 0 }
  }

  const npcs = obs.mechs.filter((m) => m.kind === "npc")
  if (npcs.length > 0) {
    let best = npcs[0]
    for (const n of npcs) if (better(best, n)) best = n
    return { id: best.id, kind: "npc", pos: best.pos, hp: best.hp ?? 0 }
  }

  if (obs.crates.length > 0) {
    let best = obs.crates[0]
    for (const c of obs.crates) if (horizDist(c.pos, self) < horizDist(best.pos, self)) best = c
    return { id: best.id, kind: "crate", pos: best.pos, hp: best.hp ?? 0 }
  }
  return null
}

/**
 * Iterative lead solve for a straight no-gravity projectile of `speed`.
 * Finds the flight time T where the bullet at T meets the target center advanced by vel*T.
 * Returns null when the target is at/over bullet speed or the iteration does not converge
 * within maxIter.
 */
export function intercept(
  muzzle: Vec3,
  target: { pos: Vec3; vel: Vec3; centerHeight: number },
  speed: number,
  maxIter = 6,
): { yaw: number; pitch: number; dist: number } | null {
  if (!(speed > 0)) return null
  const c0: Vec3 = [target.pos[0], target.pos[1], target.pos[2] + target.centerHeight]
  const v = target.vel
  if (Math.hypot(v[0], v[1], v[2]) >= speed) return null // the bullet can never catch up

  let t = dist3(muzzle, c0) / speed
  let converged = false
  for (let i = 0; i < maxIter; i++) {
    const point: Vec3 = [c0[0] + v[0] * t, c0[1] + v[1] * t, c0[2] + v[2] * t]
    const t2 = dist3(muzzle, point) / speed
    if (Math.abs(t2 - t) < 1e-5) {
      t = t2
      converged = true
      break
    }
    t = t2
  }
  if (!converged) return null

  const point: Vec3 = [c0[0] + v[0] * t, c0[1] + v[1] * t, c0[2] + v[2] * t]
  const dz = point[2] - muzzle[2]
  return {
    yaw: Math.atan2(point[1] - muzzle[1], point[0] - muzzle[0]),
    pitch: Math.atan2(dz, Math.hypot(point[0] - muzzle[0], point[1] - muzzle[1])),
    dist: dist3(muzzle, point),
  }
}

/** Low-arc launch pitch for speed v, gravity g, horizontal range R, height diff dz. Null when unreachable. */
function ballisticPitch(v: number, g: number, R: number, dz: number): number | null {
  if (!(v > 0) || !(g > 0) || !(R > 0)) return null
  const k = (v * v) / (g * R)
  const disc = k * k - 1 - (2 * v * v * dz) / (g * R * R)
  if (disc < 0) return null
  return Math.atan(k - Math.sqrt(disc))
}

/** Shot origin: SHOT_OUT along the current actual aim, pulled back to the first clear distance (min 0.5 m). */
function shotOrigin(muzzle: Vec3, aim: [number, number], los: (a: Vec3, b: Vec3) => boolean): Vec3 {
  const d = dirOf(aim[0], aim[1])
  const at = (dist: number): Vec3 => [
    muzzle[0] + d[0] * dist,
    muzzle[1] + d[1] * dist,
    muzzle[2] + d[2] * dist,
  ]
  if (los(muzzle, at(SHOT_OUT))) return at(SHOT_OUT)
  for (let dist = SHOT_OUT - 0.25; dist >= 0.5; dist -= 0.25) if (los(muzzle, at(dist))) return at(dist)
  return at(0.5)
}

export function decide(obs: Observation, state: CombatState, deps: { los: (a: Vec3, b: Vec3) => boolean }): CombatDecision | null {
  for (const ev of obs.events) {
    if (ev.type === "damaged" && ev.bearing !== null) {
      state.lastDamaged = { bearing: ev.bearing, byId: ev.by, tObsTick: obs.tick }
    }
  }

  const target = pickTarget(obs, state)
  const self = obs.self
  const gun = self.weapons.find((w) => w?.kind === "gun") ?? null

  if (!target) {
    const wantReload = gun != null && (gun.ammo ?? 0) <= 3
    // fire [false,false]: a laser mid-charge is released (fires along the current aim) rather than held
    state.laserFullTick = null
    const ld = state.lastDamaged
    if (ld && obs.tick - ld.tObsTick <= DAMAGE_WINDOW_TICKS) {
      // keep turning to the unseen attacker every decision until the 2 s window expires
      return { aim: [ld.bearing, 0], fire: [false, false], reload: wantReload, target: null }
    }
    if (ld) state.lastDamaged = null // expired
    return { aim: [self.aim[0], self.aim[1]], fire: [false, false], reload: wantReload, target: null }
  }

  const actor = obs.mechs.find((m) => m.id === target.id)
  const cubes = target.kind === "mech" ? (actor?.cubes ?? 0) : 0
  const vel: Vec3 = actor?.vel ?? [0, 0, 0]
  // aim height above feet: mech center of mass per protocol §5 (cube-scaled); half-height for NPCs, low center for crates
  const centerHeight = target.kind === "mech" ? MECH_CENTER * (1 + 0.05 * cubes) : target.kind === "npc" ? 1.75 : 1
  const center: Vec3 = [target.pos[0], target.pos[1], target.pos[2] + centerHeight]
  const muzzle: Vec3 = [self.pos[0], self.pos[1], self.pos[2] + (self.stats.muzzle_height ?? 7)]
  const horiz = Math.hypot(center[0] - muzzle[0], center[1] - muzzle[1])
  const directYaw = Math.atan2(center[1] - muzzle[1], center[0] - muzzle[0])
  const directPitch = Math.atan2(center[2] - muzzle[2], horiz)

  const radius = MECH_RADIUS * (1 + 0.05 * cubes)
  const special = self.weapons[1]
  const sol = gun ? intercept(muzzle, { pos: target.pos, vel, centerHeight }, gun.stats.speed ?? 252) : null

  // mortar lob (yaw + pitch), led by the shell's own flight time; null outside the 40–200 m window
  let lob: [number, number] | null = null
  if (special?.kind === "mortar") {
    const R0 = horizDist(target.pos, self.pos)
    if (R0 >= 40 && R0 <= 200) {
      const v = special.stats.speed ?? 72
      // mortar tube: pos.z + height + 0.5, with height cube-scaled (protocol §5)
      const tubeZ = self.pos[2] + 9 * (1 + 0.05 * self.cubes) + 0.5
      let p: Vec3 = target.pos
      for (let i = 0; i < 3; i++) {
        const R = horizDist(p, self.pos)
        const pitch = ballisticPitch(v, state.gravity, R, p[2] + centerHeight - tubeZ)
        if (pitch == null) {
          lob = null
          break
        }
        lob = [Math.atan2(p[1] - self.pos[1], p[0] - self.pos[0]), pitch]
        const T = R / (v * Math.cos(pitch))
        p = [target.pos[0] + vel[0] * T, target.pos[1] + vel[1] * T, target.pos[2] + vel[2] * T]
      }
    }
  }

  // cb/aim: a laser release is imminent once the charge passes 90% of its cap
  const laserImminent =
    special?.kind === "laser" && (special.charge ?? 0) >= 0.9 * (special.stats.charge ?? 0.75)

  // torso ownership: mortar lob when active; the laser (hitscan, no lead) takes the torso only
  // when a release is imminent -- until then the gun's lead solution keeps the gun firing;
  // without a lead solution the direct line is all we have
  let aim: [number, number]
  if (lob) aim = lob
  else if (special?.kind === "laser") {
    aim = !sol || laserImminent ? [directYaw, directPitch] : [sol.yaw, sol.pitch]
  } else if (!sol) aim = [directYaw, directPitch]
  else aim = [sol.yaw, sol.pitch]

  // gun (slot 0): fire when the actual aim is on the lead solution (yaw and pitch) and the line is
  // clear; never while the mortar owns the torso (bullets would fly along the lob)
  let gunFire = false
  if (gun && sol && !lob && !laserImminent) {
    const yawErr = Math.abs(normAngle(self.aim[0] - sol.yaw))
    const pitchErr = Math.abs(self.aim[1] - sol.pitch)
    if (yawErr <= Math.atan(radius / horiz) && pitchErr <= Math.atan(centerHeight / horiz)) {
      const aimPoint: Vec3 = [
        muzzle[0] + Math.cos(sol.pitch) * Math.cos(sol.yaw) * sol.dist,
        muzzle[1] + Math.cos(sol.pitch) * Math.sin(sol.yaw) * sol.dist,
        muzzle[2] + Math.sin(sol.pitch) * sol.dist,
      ]
      const origin = shotOrigin(muzzle, self.aim, deps.los)
      gunFire = deps.los(muzzle, aimPoint) && deps.los(origin, aimPoint)
    }
  }

  // special (slot 1): laser / lightning / mortar
  let specialFire = false
  if (special?.kind === "laser") {
    const chargeStat = special.stats.charge ?? 0.75
    const holdS = special.stats.hold_s ?? 2
    const charge = special.charge ?? 0
    const aimOk =
      Math.abs(normAngle(self.aim[0] - directYaw)) <= Math.atan(radius / horiz) &&
      Math.abs(self.aim[1] - directPitch) <= Math.atan(centerHeight / horiz)
    let release = charge >= 0.95 * chargeStat && aimOk && deps.los(muzzle, center)
    if (charge >= chargeStat - 1e-9) {
      // charge caps at `charge` (protocol §7), so time at full charge must be counted in ticks;
      // never hold past hold_s — release there regardless of aim/los
      if (state.laserFullTick == null) state.laserFullTick = obs.tick
      else if (obs.tick - state.laserFullTick >= Math.round(holdS * TICKS_PER_S)) release = true
    } else {
      state.laserFullTick = null
    }
    specialFire = !release // hold to charge; release fires the beam
  } else if (special?.kind === "lightning") {
    const range = special.stats.range ?? 45
    const halfCone = ((special.stats.cone_deg ?? 30) / 2) * (Math.PI / 180)
    const yawOff = Math.abs(normAngle(self.aim[0] - directYaw))
    // los last: don't burn energy arcing into a wall at a remembered target
    specialFire = horiz <= 0.95 * range && yawOff <= halfCone && self.energy > 15 && deps.los(muzzle, center)
  } else if (special?.kind === "mortar" && lob) {
    // shells leave along the ACTUAL torso aim: wait until it has slewed onto the lob
    const pitchErr = Math.abs(self.aim[1] - lob[1])
    const yawErr = Math.abs(normAngle(self.aim[0] - lob[0]))
    specialFire = pitchErr <= MORTAR_PITCH_TOL && yawErr <= MORTAR_PITCH_TOL
  }

  return {
    aim,
    fire: [gunFire, specialFire],
    reload: false,
    target: { id: target.id, kind: target.kind },
  }
}
