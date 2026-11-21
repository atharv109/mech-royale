/**
 * sim/house.ts — in-process "house" bots emulating the practice opponents: 15 per match
 * (4 easy / 4 medium / 4 hard / 3 expert), shared vision, never shoot each other, hunt the
 * test bot weakest-first: 1 hunter at spawn, promote another per hunter death, everyone
 * hunts in the last ~2 zone phases. Difficulty = lead, aim error, reaction, strafe, facing.
 */
import { type BotHandle } from "./engine.ts"
import { gauss } from "./config.ts"

export type Tier = "easy" | "medium" | "hard" | "expert"

export type TierKnobs = {
  lead: number // 0..1 fraction of perfect gun lead applied
  errDeg: number // aim gaussian error (deg), yaw+pitch
  reactS: number // target visible (shared) this long before firing
  strafe: boolean
  face: boolean
  fireRange: number // max gun fire distance (m)
  approach: number // hunter standoff (m)
}

export const KNOBS: Record<Tier, TierKnobs> = {
  easy: { lead: 0.5, errDeg: 6, reactS: 0.8, strafe: false, face: false, fireRange: 90, approach: 25 },
  medium: { lead: 0.75, errDeg: 3.5, reactS: 0.5, strafe: false, face: true, fireRange: 120, approach: 35 },
  hard: { lead: 0.9, errDeg: 1.9, reactS: 0.3, strafe: true, face: true, fireRange: 158, approach: 45 },
  expert: { lead: 1, errDeg: 0.95, reactS: 0.15, strafe: true, face: true, fireRange: 180, approach: 55 },
}

type HouseState = { strafePhase: number; waypoint: [number, number] | null; wpUntil: number }

export type SquadShared = {
  seen: Map<number, { pos: [number, number, number]; vel: [number, number]; hp: number; firstT: number; lastT: number; kind: string }>
  hunters: Set<number>
  deadHunters: number
  desiredHunters: number
  houseIds: number[]
  rng: () => number
  lastPhase: boolean
}

export function makeSquad(rng: () => number): SquadShared {
  return { seen: new Map(), hunters: new Set(), deadHunters: 0, desiredHunters: 1, houseIds: [], rng, lastPhase: false }
}

/** A hunter died: one more house bot should hunt. */
export function registerDeath(squad: SquadShared, id: number) {
  if (squad.hunters.has(id)) squad.deadHunters++
}

export function updateSquad(squad: SquadShared, aliveIds: number[]) {
  squad.desiredHunters = 1 + squad.deadHunters
  while (squad.hunters.size < squad.desiredHunters) {
    const pool = aliveIds.filter((id) => squad.houseIds.includes(id) && !squad.hunters.has(id))
    if (pool.length === 0) break
    squad.hunters.add(pool[Math.floor(squad.rng() * pool.length)]!)
  }
  if (squad.lastPhase) for (const id of aliveIds) if (squad.houseIds.includes(id)) squad.hunters.add(id)
}

export function createHouseBot(squad: SquadShared, tier: Tier): BotHandle {
  const st: HouseState = { strafePhase: squad.rng() * Math.PI * 2, waypoint: null, wpUntil: 0 }
  const k = KNOBS[tier]
  return {
    name: `house-${tier}`,
    act(obs: Record<string, unknown>) {
      const me = obs.self as { id: number; pos: [number, number, number]; hp: number; energy: number; in_zone: boolean; weapons: ({ kind: string; cooldown: number; ammo: number; reload: number; charge?: number; stats: Record<string, number> } | null)[] }
      const tTick = obs.tick as number
      if (!squad.houseIds.includes(me.id)) squad.houseIds.push(me.id)
      const t = obs.t as number
      const zone = obs.zone as { center: [number, number]; radius: number; next_center: [number, number]; next_radius: number; shrink_start: number }
      const mechsList = (obs.mechs as { id: number; kind: string; pos: [number, number, number]; vel: [number, number, number]; hp?: number }[]) ?? []
      // merge my view into squad memory (30 s retention)
      for (const e of mechsList) {
        const prev = squad.seen.get(e.id)
        squad.seen.set(e.id, { pos: e.pos, vel: [e.vel?.[0] ?? 0, e.vel?.[1] ?? 0], hp: e.hp ?? 1e9, firstT: prev?.firstT ?? t, lastT: t, kind: e.kind })
      }
      for (const [id, rec] of squad.seen) if (t - rec.lastT > 30) squad.seen.delete(id)

      // weakest-first target among non-house mechs
      let targetId = -1
      let target: { pos: [number, number, number]; vel: [number, number]; hp: number; lastT: number } | null = null
      for (const [id, rec] of squad.seen) {
        if (squad.houseIds.includes(id) || rec.kind !== "mech") continue
        if (!target || rec.hp < target.hp) {
          target = rec
          targetId = id
        }
      }
      const own = targetId >= 0 ? mechsList.find((e) => e.id === targetId) : undefined
      const [x, y, z] = me.pos
      const dZone = Math.hypot(x - zone.center[0], y - zone.center[1])
      const hunting = squad.hunters.has(me.id) || squad.lastPhase
      // ---------- movement ----------
      let gx = x
      let gy = y
      let boost = false
      if (!me.in_zone) {
        gx = zone.center[0]
        gy = zone.center[1]
        boost = true
      } else if (hunting && target && t - target.lastT < 12) {
        const d = Math.hypot(target.pos[0] - x, target.pos[1] - y)
        if (d > k.approach + 25) {
          gx = target.pos[0]
          gy = target.pos[1]
          boost = d > 60
        } else if (k.strafe && own) {
          const ang = Math.atan2(target.pos[1] - y, target.pos[0] - x) + Math.PI / 2
          const sdir = Math.sin(t * 2.2 + st.strafePhase)
          gx = x + Math.cos(ang) * sdir * 25
          gy = y + Math.sin(ang) * sdir * 25
        } else if (d > k.approach) {
          gx = x + (target.pos[0] - x) * 0.2
          gy = y + (target.pos[1] - y) * 0.2
        }
      } else {
        // loot nearby, else pre-position to the next circle, else roam waypoints;
        // hunters with no target chase the freshest heard sound
        const pickups = (obs.pickups as { id: number; kind: string; pos: [number, number, number] }[]) ?? []
        const cubes = (obs.cubes as { id: number; pos: [number, number, number] }[]) ?? []
        let goal: [number, number] | null = null
        let bc: { pos: [number, number, number] } | null = null
        let bd = 60
        for (const cb of cubes) {
          const d = Math.hypot(cb.pos[0] - x, cb.pos[1] - y)
          if (d < bd) {
            bd = d
            bc = cb
          }
        }
        if (bc) goal = [bc.pos[0], bc.pos[1]]
        if (!goal) {
          let bp: { pos: [number, number, number] } | null = null
          let bd2 = 50
          for (const pu of pickups) {
            if (pu.kind === "gun") continue
            const d = Math.hypot(pu.pos[0] - x, pu.pos[1] - y)
            if (d < bd2) {
              bd2 = d
              bp = pu
            }
          }
          if (bp) goal = [bp.pos[0], bp.pos[1]]
        }
        if (!goal) {
          const dNext = Math.hypot(x - zone.next_center[0], y - zone.next_center[1]) - zone.next_radius
          if (dNext > 0 && zone.shrink_start - t <= dNext / 11 + 10) {
            goal = [zone.next_center[0], zone.next_center[1]]
            boost = dNext > 100
          } else if (dZone > zone.radius * 0.95) {
            goal = [x + (zone.center[0] - x) * 0.3, y + (zone.center[1] - y) * 0.3]
          } else {
            const snds = (obs.sounds as { kind: string; bearing: number; age: number }[]) ?? []
            let best: { bearing: number; age: number } | null = null
            for (const sd of snds) {
              if (["gun", "laser", "lightning", "mortar", "explosion", "step", "boost", "jet", "reload"].includes(sd.kind) && sd.age < 4) {
                if (!best || sd.age < best.age) best = sd
              }
            }
            if (best) goal = [x + Math.cos(best.bearing) * 60, y + Math.sin(best.bearing) * 60]
          }
          if (!goal) {
            if (st.wpUntil < t || !st.waypoint) {
              st.wpUntil = t + 8 + squad.rng() * 6
              const a = squad.rng() * Math.PI * 2
              const rr = (0.15 + 0.8 * squad.rng()) * zone.radius
              st.waypoint = [zone.center[0] + Math.cos(a) * rr, zone.center[1] + Math.sin(a) * rr]
            }
            goal = st.waypoint
            if (goal && Math.hypot(goal[0] - x, goal[1] - y) > 60) boost = true
          }
        }
        if (goal) {
          gx = goal[0]
          gy = goal[1]
        }
      }
      let mx = gx - x
      let my = gy - y
      const ml0 = Math.hypot(mx, my)
      if (k.strafe && own) {
        const ang = Math.atan2(own.pos[1] - y, own.pos[0] - x) + Math.PI / 2
        const sd2 = Math.sin(t * 2.2 + st.strafePhase) * 0.8
        mx = mx / (ml0 || 1) + Math.cos(ang) * sd2
        my = my / (ml0 || 1) + Math.sin(ang) * sd2
      }
      const ml2 = Math.hypot(mx, my) || 1
      const move: [number, number] = [mx / ml2, my / ml2]
      if (ml0 < 2) {
        move[0] = 0
        move[1] = 0
      }
      // ---------- aiming ----------
      const dTgt = own ? Math.hypot(own.pos[0] - x, own.pos[1] - y) : 0
      const react = target && t - (target as { firstT: number }).firstT >= k.reactS ? true : false
      let yawErr = 0
      let pitchErr = 0
      if (react) {
        yawErr = gauss(squad.rng) * (k.errDeg * Math.PI) / 180
        pitchErr = gauss(squad.rng) * (k.errDeg * Math.PI) / 180
      }
      let aim: [number, number]
      let fire: [boolean, boolean] = [false, false]
      if (own && react) {
        // gun: lead the target along its velocity, then add gaussian error
        const w1 = me.weapons[0]
        const speed = w1?.stats?.speed ?? 252
        const tof1 = dTgt / speed
        const tof2 = Math.hypot(own.pos[0] + own.vel[0] * tof1 * k.lead - x, own.pos[1] + own.vel[1] * tof1 * k.lead - y) / speed
        const tx = own.pos[0] + own.vel[0] * tof2 * k.lead
        const ty = own.pos[1] + own.vel[1] * tof2 * k.lead
        aim = [Math.atan2(ty - y, tx - x) + yawErr, Math.atan2(own.pos[2] + 4.5 - (z + 7), dTgt) + pitchErr]
        fire[0] = dTgt <= k.fireRange
      } else {
        // face where we're heading while roaming (wedge aligned with movement);
        // when standing, sweep a slow search rotation
        aim = [ml0 > 2 ? Math.atan2(my, mx) : t * 0.9 + st.strafePhase, 0]
      }
      // special weapon (slot 1) overrides aim + fire
      const w2 = me.weapons[1]
      if (own && react && w2) {
        const d2 = Math.hypot(own.pos[0] - x, own.pos[1] - y)
        const yaw0 = Math.atan2(own.pos[1] - y, own.pos[0] - x)
        if (w2.kind === "laser") {
          if (d2 <= (w2.stats.range ?? 280) * 0.85) {
            // hold to charge, release once charged
            fire[1] = (w2.charge ?? 0) < (w2.stats.charge ?? 0.75) - 0.02
            aim = [yaw0 + yawErr, Math.atan2(own.pos[2] + 4.5 - (z + 8), d2) + pitchErr]
          } else {
            fire[1] = false
            aim = [yaw0, Math.atan2(own.pos[2] + 4.5 - (z + 8), d2)]
          }
        } else if (w2.kind === "lightning") {
          fire[1] = d2 <= 38 && me.energy >= 10
          aim = [yaw0 + yawErr, 0]
        } else if (w2.kind === "mortar") {
          if (d2 >= 40 && d2 <= 210) {
            const v2 = w2.stats.speed ?? 72
            const pitch = 0.5 * Math.asin(Math.min(1, (d2 * 22) / (v2 * v2)))
            aim = [yaw0 + yawErr, pitch]
            fire[1] = true
          } else {
            fire[1] = false
            aim = [yaw0, 0]
          }
        }
      }
      // pick up a free-slot special within 3.5 m
      let take: number | undefined
      const pus = (obs.pickups as { id: number; kind: string; pos: [number, number, number] }[]) ?? []
      for (const pu of pus) {
        if (pu.kind === "gun") continue
        if (Math.hypot(pu.pos[0] - x, pu.pos[1] - y) <= 3.5 && !me.weapons.some((w) => w?.kind === pu.kind)) {
          take = pu.id
          break
        }
      }
      return {
        tick: tTick,
        move,
        aim,
        fire,
        boost: boost && me.energy > 25,
        ...(take !== undefined ? { take } : {}),
        ...(obs.offer ? { pick: 1 } : {}),
      }
    },
  }
}
