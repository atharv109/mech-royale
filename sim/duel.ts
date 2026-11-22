/** sim/duel.ts - Task 13 cb/both-duel: 1v1 in the sim arena vs one house bot.
 * Usage: bun sim/duel.ts [--bot <path-to-bundle|bot.ts>] [--trials 20] [--tiers hard,expert]
 * Prints per-tier win/loss/draw and median win time. Read-only diagnostic.
 */
import { Engine } from "./engine.ts"
import { loadHello } from "./config.ts"
import { loadMap, feasible } from "../map.ts"
import { makeSquad, updateSquad, registerDeath, createHouseBot } from "./house.ts"
import { spawnBot, decide as botDecide, endBot } from "./botio.ts"

const hello = loadHello()
const CFG = hello.config
const map = loadMap("C:/Users/athar/OneDrive/Desktop/mech royale/kit/arena", CFG.mech.step_up)

function findPair(distWant: number, seedProbe: number): [[number, number], [number, number]] {
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

type Tier = "hard" | "expert"
type Outcome = { res: "win" | "loss" | "draw"; t: number; hp: number; oppHp: number; shots: number }

async function trial(tier: Tier, seedN: number, cmd: string[]): Promise<Outcome> {
  const eng = new Engine(CFG, map, seedN)
  for (let i = 2; i < eng.mechs.length; i++) eng.mechs[i]!.alive = false
  for (const n of eng.npcs) n.alive = false
  let s = seedN >>> 0
  const srng = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
  const squad = makeSquad(srng)
  squad.houseIds = [1]
  const oppBot = createHouseBot(squad, tier)
  const distWant = tier === "hard" ? 70 : 90
  const [a, b] = findPair(distWant, seedN)
  const me = eng.mechs[0]!
  const opp = eng.mechs[1]!
  me.pos = [a[0], a[1], me.pos[2]]
  opp.pos = [b[0], b[1], opp.pos[2]]
  me.hp = me.maxHp
  opp.hp = opp.maxHp
  const sub = await spawnBot(cmd, ".", { ...hello, slot: 0, seed: seedN, map: { ...hello.map, dir: "C:/Users/athar/OneDrive/Desktop/mech royale/kit/arena" } })
  let prevOppAlive = true
  let endT = 0
  while (!eng.finished() && eng.t < 150 && me.alive && opp.alive) {
    updateSquad(squad, [0, 1])
    for (const id of [1, 0] as const) {
      const m = eng.mechs[id]!
      if (!m.alive) continue
      const obs = eng.buildObs(m)
      if (id === 0) {
        const got = await botDecide(sub, obs, CFG.bot.tick_budget_ms)
        eng.applyAction(m, (got ?? {}) as Record<string, unknown>)
      } else {
        eng.applyAction(m, oppBot.act(obs) as Record<string, unknown>)
      }
    }
    eng.step(eng.dt)
    eng.step(eng.dt)
    if (!opp.alive && prevOppAlive) registerDeath(squad, 1)
    prevOppAlive = opp.alive
  }
  endT = eng.t
  const res = !opp.alive && me.alive ? "win" : !me.alive && opp.alive ? "loss" : "draw"
  const out: Outcome = { res, t: endT, hp: Math.round(me.hp), oppHp: Math.round(opp.hp), shots: eng.shotsByFocus }
  await endBot(sub, me.alive ? 1 : 16, endT)
  return out
}

async function main() {
  const args = process.argv.slice(2)
  let botPath = "bot.ts"
  let trials = 20
  let tiers: Tier[] = ["hard", "expert"]
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--bot") botPath = args[++i]!
    else if (args[i] === "--trials") trials = parseInt(args[++i]!)
    else if (args[i] === "--tiers") tiers = args[++i]!.split(",") as Tier[]
  }
  const med = (xs: number[]) => (xs.length ? xs.slice().sort((a2, b2) => a2 - b2)[Math.floor(xs.length / 2)]! : 0)
  for (const tier of tiers) {
    const wins: number[] = []
    let loss = 0, draw = 0, shotsTot = 0, dmgDealt = 0
    for (let i = 0; i < trials; i++) {
      const r = await trial(tier, 3000 + i * 13, ["bun", botPath])
      if (r.res === "win") wins.push(r.t)
      else if (r.res === "loss") loss++
      else draw++
      shotsTot += r.shots
      dmgDealt += r.oppHp > 0 ? 200 - r.oppHp : 200
    }
    const pct = (100 * wins.length) / trials
    console.log(`${tier}: winRate=${pct.toFixed(0)}% (${wins.length}W/${loss}L/${draw}D of ${trials}) ` +
      `medianWin=${wins.length ? med(wins).toFixed(1) : "-"}s avgShots=${(shotsTot / trials).toFixed(0)}`)
  }
}
await main()
