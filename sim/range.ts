/** sim/range.ts — Task 8 combat diagnostic: v1 vs one scripted target, 20 trials per condition.
 * Metrics: gun shots/s while target visible, hit rate, time to kill, and a no-fire breakdown
 * by gate (replicating combat.ts math on the bot's own observations). Read-only diagnosis;
 * no engine/bot code touched. Run: bun sim/range.ts
 */
import { Engine } from "./engine.ts"
import { loadHello } from "./config.ts"
import { loadMap, los, feasible } from "../map.ts"
import { decide, type CombatState, intercept } from "../combat.ts"

const normAngle = (a: number): number => {
  a %= 2 * Math.PI
  if (a > Math.PI) a -= 2 * Math.PI
  else if (a < -Math.PI) a += 2 * Math.PI
  return a
}
const SHOT_OUT = 3
const dirOf = (yaw: number, pitch: number): [number, number, number] => {
  const cp = Math.cos(pitch)
  return [cp * Math.cos(yaw), cp * Math.sin(yaw), Math.sin(pitch)]
}
const shotOrigin = (muzzle: [number, number, number], aim: [number, number], losF: (a: Vec3Like, b: Vec3Like) => boolean): [number, number, number] => {
  const d = dirOf(aim[0], aim[1])
  const at = (dist: number): [number, number, number] => [muzzle[0] + d[0] * dist, muzzle[1] + d[1] * dist, muzzle[2] + d[2] * dist]
  if (losF(muzzle, at(SHOT_OUT))) return at(SHOT_OUT)
  for (let dist = SHOT_OUT - 0.25; dist >= 0.5; dist -= 0.25) if (losF(muzzle, at(dist))) return at(dist)
  return at(0.5)
}
type Vec3Like = [number, number, number]
import { spawnBot, decide as botDecide, endBot, type SubBot } from "./botio.ts"

const hello = loadHello()
const CFG = hello.config
const map = loadMap("C:/Users/athar/OneDrive/Desktop/mech royale/kit/arena", CFG.mech.step_up)
const R0 = CFG.mech.radius || 2.5

function findPair(distWant: number, seedProbe = 1): [[number, number], [number, number]] {
  const probe = new Engine(CFG, map, seedProbe)
  const zc = CFG.zone.start_center
  const Rz = CFG.zone.start_radius
  for (let mx = -Rz; mx <= Rz; mx += 8) {
    for (let my = -Rz; my <= Rz; my += 8) {
      const ax = zc[0] + mx, ay = zc[1] + my
      if (!feasible(map, ax, ay)) continue
      for (let k = 0; k < 16; k++) {
        const ang = (k / 16) * Math.PI * 2
        const bx = ax + Math.cos(ang) * distWant, by = ay + Math.sin(ang) * distWant
        if (!feasible(map, bx, by)) continue
        if (Math.hypot(bx - zc[0], by - zc[1]) > Rz - 10) continue
        probe.mechs[0]!.pos = [ax, ay, 0]
        probe.mechs[1]!.pos = [bx, by, 0]
        probe.mechs[0]!.aim = [ang, 0]
        if (probe.canSee(probe.mechs[0]!, [bx, by, 4.5], 9)) return [[ax, ay], [bx, by]]
      }
    }
  }
  throw new Error(`no open pair at ${distWant} m`)
}

type Cond = "still60" | "strafe60" | "strafe120"

async function trial(cond: Cond, seedN: number) {
  const eng = new Engine(CFG, map, seedN)
  for (let i = 2; i < eng.mechs.length; i++) eng.mechs[i]!.alive = false
  for (const n of eng.npcs) n.alive = false
  const distWant = cond === "strafe120" ? 120 : 60
  const [a, b] = findPair(distWant)
  const me = eng.mechs[0]!
  const tgt = eng.mechs[1]!
  me.pos = [a[0], a[1], me.pos[2]]
  tgt.pos = [b[0], b[1], tgt.pos[2]]
  tgt.hp = tgt.maxHp

  const sub = await spawnBot(["bun", "bot.ts"], ".", { ...hello, slot: 0, seed: seedN, map: { ...hello.map, dir: "C:/Users/athar/OneDrive/Desktop/mech royale/kit/arena" } })

  // monkey-patch damage for hit counting (instance shadow; engine code untouched)
  let gunHits = 0
  const origDamage = eng.damage.bind(eng)
  ;(eng as unknown as { damage: unknown }).damage = (victim: unknown, amount: number, info: { byId: number; weapon: string }) => {
    const v = victim as { id: number }
    if (v.id === 1 && info.byId === 0 && info.weapon === "gun") gunHits++
    return origDamage(victim as Parameters<typeof origDamage>[0], amount, info as Parameters<typeof origDamage>[2])
  }

  let shots = 0
  let prevFocusShots = eng.shotsByFocus
  const gates = { fired: 0, notVisible: 0, aimGate: 0, los: 0, cooldown: 0, other: 0 }
  let visibleDecisions = 0
  let killT: number | null = null
  let slewStreak = 0
  const slewStreaks: number[] = []
  const combat: CombatState = { gravity: CFG.weapons?.mortar?.gravity ?? 22, lastDamaged: null, disengaging: false }
  let projCount = eng.projs.length
  let flipAt = 1.5
  let sdir = 1

  while (!eng.finished() && eng.t < 60 && killT === null) {
    if (cond === "still60") {
      eng.applyAction(tgt, { move: [0, 0], aim: [Math.atan2(a[1] - b[1], a[0] - b[0]), 0] })
    } else {
      if (eng.t >= flipAt) { flipAt = eng.t + 1.5; sdir = -sdir }
      const perp = Math.atan2(b[1] - a[1], b[0] - a[0]) + Math.PI / 2
      eng.applyAction(tgt, { move: [Math.cos(perp) * sdir, Math.sin(perp) * sdir] })
    }
    const obs = eng.buildObs(me)
    // Task 13: the base brain ignores mechs for its first 75 s; shift the observed clock so
    // the range exercises the combat layer (engine time stays real, drops never interfere)
    obs.t = (obs.t as number) + 80
    obs.tick = (obs.tick as number) + 1600
    // Task 13: the base brain ignores mechs for its first 75 s; shift the observed clock so
    // the range exercises the combat layer (engine time stays real, drops never interfere)
    obs.t = (obs.t as number) + 80
    obs.tick = (obs.tick as number) + 1600
    const hasTarget = obs.mechs.some((m) => m.id === 1)
    if (hasTarget) visibleDecisions++
    const wPre = me.weapons[0]
    const cdPre = wPre ? wPre.cooldown > 1e-6 || (wPre.ammo ?? 0) === 0 || wPre.reload > 0 : false
    const action = (await botDecide(sub, obs, CFG.bot.tick_budget_ms)) ?? {}
    eng.applyAction(me, action)
    eng.step(eng.dt)
    eng.step(eng.dt)

    const fired0 = eng.shotsByFocus - prevFocusShots
    prevFocusShots = eng.shotsByFocus
    shots += fired0
    if (fired0 > 0) gates.fired++

    // classify the decision that just happened
    const w0 = me.weapons[0]
    if (!hasTarget) {
      if (fired0 === 0) gates.notVisible++
    } else if (fired0 > 0) {
      // fired
    } else if (cdPre) {
      gates.cooldown++
      slewStreak = 0
    } else if (action.fire?.[0] && fired0 === 0) {
      gates.other++ // bot sent fire, engine refused while off cooldown: investigate
    } else {
      // no fire intent: aim gate vs LOS vs other (replicate combat math on same obs)
      const actor = obs.mechs.find((m) => m.id === 1)!
      const muzzle = [me.pos[0], me.pos[1], me.pos[2] + (me.stats.muzzle_height ?? 7)]
      const sol = w0 ? intercept(muzzle, { pos: actor.pos, vel: actor.vel, centerHeight: 4.5 }, w0.stats?.speed ?? 252) : null
      if (!w0 || !sol) gates.other++
      else {
        const horiz = Math.hypot(actor.pos[0] - me.pos[0], actor.pos[1] - me.pos[1]) || 1
        const yawErr = Math.abs(normAngle(me.aim[0] - sol.yaw))
        const pitchErr = Math.abs(me.aim[1] - sol.pitch)
        const aimOk = yawErr <= Math.atan(R0 / horiz) && pitchErr <= Math.atan(4.5 / horiz)
        if (!aimOk) {
          gates.aimGate++
          slewStreak++
        } else {
          if (slewStreak > 0) { slewStreaks.push(slewStreak); slewStreak = 0 }
          const aimP: [number, number, number] = [
            muzzle[0] + Math.cos(sol.pitch) * Math.cos(sol.yaw) * sol.dist,
            muzzle[1] + Math.cos(sol.pitch) * Math.sin(sol.yaw) * sol.dist,
            muzzle[2] + Math.sin(sol.pitch) * sol.dist,
          ]
          const origin = shotOrigin(muzzle, me.aim, (x, y) => los(map, x, y))
          if (!(los(map, muzzle, aimP) && los(map, origin, aimP))) gates.los++
          else gates.other++
        }
      }
    }
    if (tgt.hp <= 0 && killT === null) killT = eng.t
  }
  if (slewStreak > 0) slewStreaks.push(slewStreak)
  await endBot(sub, killT ? 1 : 16, 0)
  return { shots, gunHits, visibleDecisions, killT, gates: { ...gates }, slewStreaks }
}

async function main() {
  const conds: Cond[] = ["still60", "strafe60", "strafe120"]
  const med = (xs: number[]) => (xs.length ? xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]! : 0)
  for (const cond of conds) {
    const agg = { shots: 0, gunHits: 0, visibleDecisions: 0, fired: 0, notVisible: 0, aimGate: 0, los: 0, cooldown: 0, other: 0 }
    const ttks: number[] = []
    const slews: number[] = []
    for (let i = 0; i < 20; i++) {
      const r = await trial(cond, 1000 + i * 7)
      agg.shots += r.shots
      agg.gunHits += r.gunHits
      agg.visibleDecisions += r.visibleDecisions
      agg.fired += r.gates.fired
      agg.notVisible += r.gates.notVisible
      agg.aimGate += r.gates.aimGate
      agg.los += r.gates.los
      agg.cooldown += r.gates.cooldown
      agg.other += r.gates.other
      if (r.killT !== null) ttks.push(r.killT)
      for (const s of r.slewStreaks) slews.push(s)
    }
    const visSec = agg.visibleDecisions * CFG.match.decide_every * 0.05
    const rate = visSec > 0 ? agg.shots / visSec : 0
    const hitRate = agg.shots > 0 ? (100 * agg.gunHits) / agg.shots : 0
    const visTotal = agg.visibleDecisions
    const pct = (x: number) => (visTotal ? ((100 * x) / visTotal).toFixed(1) : "0")
    console.log(`${cond}: shots/s=${rate.toFixed(2)} (max 4.17) hitRate=${hitRate.toFixed(0)}% ttk=${ttks.length ? med(ttks).toFixed(1) : "DNF"}s slain=${ttks.length}/20`)
    console.log(`   no-fire%: notVis=${pct(agg.notVisible)} aimGate=${pct(agg.aimGate)} los=${pct(agg.los)} cooldown=${pct(agg.cooldown)} other=${pct(agg.other)} fired=${pct(agg.fired)} | slewStreak median=${med(slews)} ticks (${(med(slews) * 0.1).toFixed(2)} s)`)
  }
}
await main()
