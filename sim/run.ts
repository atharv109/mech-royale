/**
 * sim/run.ts — CLI: bun sim/run.ts --bot "bun bot.ts" --matches 48 --seed 1 --jobs 4
 * Also: --starter (score kit/python/example_bot.py), --worker (internal), --replays K,
 * --quiet. Aggregates per-match results into a summary table.
 */
import { loadHello, arenaDir, rng } from "./config.ts"
import { Engine, type BotHandle, type MatchResult } from "./engine.ts"
import { makeSquad, updateSquad, registerDeath, createHouseBot, type Tier, type SquadShared } from "./house.ts"
import { spawnBot, decide, coastOf, endBot, type SubBot } from "./botio.ts"
import { loadMap, type GameMap } from "../map.ts"
import { writeFileSync, mkdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

const hello = loadHello()
const CFG = hello.config

type Args = Record<string, string | number | boolean>
function parseArgs(argv: string[]): Args {
  const a: Args = {}
  for (let i = 2; i < argv.length; i++) {
    const s = argv[i]!
    if (s.startsWith("--")) {
      const key = s.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith("--")) a[key] = true
      else {
        a[key] = key === "matches" || key === "jobs" || key === "replays" || key === "jobs" || key === "seed" ? Number(next) : next
        if (a[key] === (true as never)) a[key] = true
        i++
      }
    }
  }
  return a
}

function shlex(s: string): string[] {
  const out: string[] = []
  for (const tok of s.match(/(?:[^\s"]+|"[^"]*")+/g) ?? []) {
    out.push(tok.startsWith('"') ? tok.slice(1, -1) : tok)
  }
  return out
}

const TIERS: Tier[] = ["easy", "medium", "hard", "expert"]

// ---------- one match ----------

export async function runMatch(
  map: GameMap,
  matchSeed: number,
  testBot: BotHandle,
  opts: { replayPath?: string; quiet?: boolean },
): Promise<MatchResult> {
  const eng = new Engine(CFG, map, matchSeed)
  const squad: SquadShared = makeSquad(rng(matchSeed + 1013904223))
  // shuffle tiers deterministically: 4 easy, 4 medium, 4 hard, 3 expert
  const tierList: Tier[] = ["easy", "easy", "easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard", "expert", "expert", "expert"]
  for (let i = tierList.length - 1; i > 0; i--) {
    const j = Math.floor(squad.rng() * (i + 1))
    ;[tierList[i], tierList[j]] = [tierList[j]!, tierList[i]!]
  }
  const bots: BotHandle[] = []
  let sub: SubBot | null = null
  if (testBot.cmd) {
    sub = await spawnBot(testBot.cmd, testBot.cwd, { ...hello, slot: 0, seed: (matchSeed ^ 0x9e3779b9) >>> 0, map: { ...hello.map, dir: arenaDir() } })
    bots[0] = { name: sub.name }
    if (!opts.quiet) console.log(`test bot ready: ${sub.name}`)
  } else bots[0] = testBot
  for (let i = 1; i < 16; i++) bots[i] = createHouseBot(squad, tierList[i - 1]!)
  // tier of each house bot
  eng.names[0] = sub ? sub.name : "test-bot"
  for (let i = 1; i < 16; i++) eng.names[i] = tierList[i - 1]!
  const decideEvery = CFG.match.decide_every
  const budget = CFG.bot.tick_budget_ms
  let prevAlive = eng.mechs.map((m) => m.alive)
  const t0 = performance.now()
  while (!eng.finished()) {
    // hunter bookkeeping
    const deadNow: number[] = []
    prevAlive.forEach((was, id) => {
      if (was && !eng.mechs[id]!.alive) deadNow.push(id)
    })
    prevAlive = eng.mechs.map((m) => m.alive)
    for (const id of deadNow) registerDeath(squad, id)
    squad.lastPhase = eng.zonePhase >= CFG.zone.phases.length - 2
    updateSquad(squad, eng.mechs.map((m) => m.alive && m.place === undefined ? -1 : m.id).filter((id) => id >= 0))
    // decisions
    for (let id = 0; id < 16; id++) {
      const m = eng.mechs[id]!
      if (!m.alive) continue
      const obs = eng.buildObs(m)
      let action: Record<string, unknown> | null = null
      const bot = bots[id]!
      if (id === 0 && sub) {
        const got = await decide(sub, obs, budget)
        if (got) {
          sub.lastAnswerT = eng.t
          sub.lastAction = got
          action = got
        } else if (eng.t - sub.lastAnswerT > CFG.bot.coast_s) action = {}
        else action = coastOf(sub.lastAction)
      } else if (bot.act) {
        action = bot.act(obs) as Record<string, unknown>
      } else action = {}
      if (!action) action = {}
      eng.applyAction(m, action)
    }
    for (let k = 0; k < decideEvery; k++) {
      eng.step(eng.dt)
    }
  }
  // end for subprocess
  const res = eng.result(0)
  let bsig: Record<string, unknown> | null = null
  if (sub) {
    const f = res.placements.find((p) => p.id === 0)
    bsig = await endBot(sub, f?.place ?? 16, f?.kills ?? 0)
    ;(res as unknown as { botSig?: unknown }).botSig = bsig
  }
  if (opts.replayPath) writeFileSync(opts.replayPath, JSON.stringify(replayFile(eng)), null)
  void t0
  return res
}

// ---------- replay file ----------

function replayFile(eng: Engine): Record<string, unknown> {
  const prisms = worldPrisms()
  return { frames: eng.replay, world: { prisms } }
}

let cachedPrisms: unknown = null
function worldPrisms(): unknown {
  if (cachedPrisms) return cachedPrisms
  const world = JSON.parse(readFileSync(join(arenaDir(), "v2/world.json"), "utf8")) as { prisms: { kind: string; base: number; top: number; ring: number[] }[] }
  cachedPrisms = (world.prisms ?? []).map((p) => ({ k: p.kind, b: Math.round(p.base * 10) / 10, t: Math.round(p.top * 10) / 10, r: p.ring.map((v) => Math.round(v * 10) / 10) }))
  return cachedPrisms
}

// ---------- driver ----------

type MatchRow = {

  seed: number
  score: number
  place: number
  deathT: number | null
  cause: { by: number | null; weapon: string } | null
  dmgTaken: Record<string, number>
  dmgTakenFront: number
  dmgTakenRear: number
  dmgDealt: number
  kills: number
  survived: boolean
  duration: number
  ms: number
  aliveAtDeath: number
  hp: number
  winnerName?: string
}

async function runOne(map: GameMap, seed: number, testBot: BotHandle, matchesDone: number, replays: number, dir: string, quiet: boolean): Promise<MatchRow> {
  const botName = testBot.cmd ? testBot.cmd.join(" ") : "in-process"
  const ms0 = performance.now()
  const res = await runMatch(map, seed, testBot, { quiet, replayPath: matchesDone < replays ? join(dir, `m-${seed}.json`) : undefined })
  const bsig = (res as unknown as { botSig?: Record<string, unknown> | null }).botSig ?? null
  const ms = Math.round(performance.now() - ms0)
  const f = res.focus
  if (!f) throw new Error("no focus")
  const winner = res.placements[0]?.name
  const wep = Math.round((f.dmgTaken.gun ?? 0) + (f.dmgTaken.laser ?? 0) + (f.dmgTaken.lightning ?? 0) + (f.dmgTaken.mortar ?? 0))
  if (!quiet) console.log(`seed ${seed}: score ${f.score} place ${f.place} ${f.survived ? "survived" : `died ${(f.deathT ?? 0).toFixed(0)}s ${f.cause?.weapon ?? "?"}${f.cause?.by !== null && f.cause?.by !== undefined ? `(by ${f.cause.by})` : ""}`} alive@death ${f.aliveAtDeath ?? 1} hp-left ${Math.round(f.hp)} ${ms}ms`)
  return {
    seed,
    score: f.score,
    place: f.place,
    deathT: f.deathT,
    cause: f.cause,
    dmgTaken: f.dmgTaken,
    dmgTakenFront: f.dmgTakenFront,
    dmgTakenRear: f.dmgTakenRear,
    dmgDealt: f.dmgDealt,
    kills: f.kills,
    survived: f.survived,
    duration: res.duration,
    ms,
    aliveAtDeath: (f as unknown as { aliveAtDeath?: number }).aliveAtDeath ?? 1,
    hp: (f as unknown as { hp?: number }).hp ?? 0,
    sigCubesAt90: (f as unknown as { sigCubesAt90?: number }).sigCubesAt90 ?? 0,
    sigSpecialsAt90: (f as unknown as { sigSpecialsAt90?: number }).sigSpecialsAt90 ?? 0,
    sigCrates: (res as unknown as { sigCrates?: number }).sigCrates ?? 0,
    sigDrones: (res as unknown as { sigDrones?: number }).sigDrones ?? 0,
    sigDrops: (res as unknown as { sigDrops?: number }).sigDrops ?? 0,
    sigCubesCollected: (res as unknown as { sigCubesCollected?: number }).sigCubesCollected ?? 0,
    sigDodges: (f as unknown as Record<string, number>).sigDodges,
    sigCoverFrac: (res as unknown as Record<string, number>).sigCoverFrac,
    botDodges: typeof bsig?.dodges === "number" ? bsig.dodges : 0,
    botRetreats: typeof bsig?.retreats === "number" ? bsig.retreats : 0,
    botMaxActMs: typeof bsig?.maxActMs === "number" ? Math.round(bsig.maxActMs * 100) / 100 : 0,
    botSlowTicks: typeof bsig?.slowTicks === "number" ? bsig.slowTicks : 0,
    botParts: bsig?.slowSnap ?? null,
    botSeen: bsig?.seen ?? {},
    botSeenDist: bsig?.seenDist ?? {},
    botFail: bsig?.fail ?? {},
    botTook: bsig?.took ?? {},
    botWalked: typeof bsig?.walked === "number" ? bsig.walked : 0,
    botCoverTicks: typeof bsig?.coverTicks === "number" ? Math.round(bsig.coverTicks) : 0,
    winnerName: winner,
  }
}

async function runBatch(map: GameHandle, count: number, seed0: number, testBot: BotHandle, replays: number, dir: string, quiet: boolean): Promise<MatchRow[]> {
  const rows: MatchRow[] = []
  for (let i = 0; i < count; i++) {
    const seed = seed0 + i
    rows.push(await runOne(map, seed, testBot, i, replays, dir, quiet))
  }
  return rows
}

// map cache per process
let _map: GameMap | null = null
function getMap(): GameMap {
  _map ??= loadMap(arenaDir(), CFG.mech.step_up ?? 1.2)
  return _map
}

function summary(rows: MatchRow[]): string {
  const n = rows.length
  const mean = rows.reduce((s, r) => s + r.score, 0) / n
  const alive = rows.filter((r) => r.survived).length
  const deathTimes = rows.filter((r) => r.deathT !== null).map((r) => r.deathT!).sort((x, y) => x - y)
  const medT = deathTimes.length ? deathTimes[deathTimes.length >> 1] : null
  const causes: Record<string, number> = {}
  let front = 0
  let rear = 0
  for (const r of rows) {
    const key = r.cause ? (r.cause.weapon !== "zone" ? `${r.cause.weapon}` : "zone") : "survived"
    causes[key] = (causes[key] ?? 0) + 1
    front += r.dmgTakenFront
    rear += r.dmgTakenRear
  }
  const dmg: Record<string, number> = {}
  for (const r of rows) for (const [k, v] of Object.entries(r.dmgTaken)) dmg[k] = (dmg[k] ?? 0) + v
  const lines = [
    `matches ${n} | mean score ${mean.toFixed(2)} | survived ${alive}/${n} | median death ${medT === null ? "-" : `${medT.toFixed(0)}s of 480s`}`,
    `death causes: ${Object.entries(causes).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${v}`).join(", ") || "-"}`,
    `damage taken by weapon: ${Object.entries(dmg).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${Math.round(v)}`).join(", ")}`,
    `front/rear total: ${Math.round(front)} / ${Math.round(rear)} (rear ${Math.round((100 * rear) / Math.max(1, front + rear))}%)`,
  ]
  return lines.join("\n")
}

type GameHandle = GameMap

async function main() {
  const a = parseArgs(process.argv)
  const dir = join(import.meta.dir, "replays")
  mkdirSync(dir, { recursive: true })
  const replays = (a.replays as number) ?? 2
  const matches = (a.matches as number) ?? 48
  const jobs = (a.jobs as number) ?? 1
  const seed = (a.seed as number) ?? 1
  const quiet = !!a.quiet
  let testBot: BotHandle
  if (a.starter) {
    testBot = { cmd: [process.env.PY || "C:/Users/athar/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/python.exe", "-u", "example_bot.py"], cwd: join(import.meta.dir, "../../kit/python") }
  } else if (a.bot) {
    testBot = { cmd: shlex(a.bot as string), cwd: join(import.meta.dir, "..") }
  } else {
    testBot = { act: () => ({}) }
  }
  if (a.worker) {
    const rows = await runBatch(getMap(), matches, seed, testBot, replays, dir, quiet)
    for (const r of rows) console.log("ROW " + JSON.stringify(r))
    return
  }
  const t0 = performance.now()
  let allRows: MatchRow[] = []
  if (jobs <= 1) {
    allRows = await runBatch(getMap(), matches, seed, testBot, replays, dir, quiet)
  } else {
    const per = Math.ceil(matches / jobs)
    const procs: Promise<MatchRow[]>[] = []
    for (let j = 0; j < jobs; j++) {
      const count = Math.min(per, matches - j * per)
      if (count <= 0) break
      const argv = ["bun", join(import.meta.dir, "run.ts"), "--worker", "--matches", String(count), "--seed", String(seed + j * per), "--replays", String(replays), ...(a.starter ? ["--starter"] : a.bot ? ["--bot", a.bot as string] : []), ...(quiet ? ["--quiet"] : [])]
      procs.push(
        (async () => {
          const p = Bun.spawn({ cmd: argv, cwd: join(import.meta.dir, ".."), stdout: "pipe", stderr: "inherit" })
          const out = await new Response(p.stdout).text()
          await p.exited
          return out.split("\n").filter((l) => l.startsWith("ROW ")).map((l) => JSON.parse(l.slice(4)) as MatchRow)
        })()
      )
    }
    const parts = await Promise.all(procs)
    allRows = parts.flat()
  }
  const totalMs = Math.round(performance.now() - t0)
  // per-match table
  for (const r of allRows.sort((x, y) => x.seed - y.seed)) console.log("ROW " + JSON.stringify(r))
  console.log("\n== summary ==")
  console.log(summary(allRows))
  console.log(`\n${matches} matches in ${(totalMs / 1000).toFixed(1)} s (${(totalMs / matches / 1000).toFixed(1)} s/match, jobs ${jobs})`)
}

await main()
