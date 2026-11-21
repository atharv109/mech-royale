/**
 * stress.ts — Task 3 timing stress: replay the recorded match stream through brain.act()
 * with a forced A* replan on every act, and report act() wall-time percentiles.
 *
 *   bun scripts/stress.ts               (REPLAYS=5 by default)
 *   STRESS_REPLAYS=8 bun scripts/stress.ts
 *   ARENA_DIR=/arena bun scripts/stress.ts   (sandboxed docker run: mount the kit's arena)
 *
 * Forcing every tick to replan is the worst case for the 50 ms tick budget; the act() deadline
 * (A* only while act < 10 ms elapsed, cap dropped to 10 ms) is active throughout. Target: max
 * act() < 35 ms on 1 CPU.
 */
import { readFileSync } from "node:fs"
import { gunzipSync } from "node:zlib"
import { join } from "node:path"

import { Brain } from "../brain.ts"
import { loadMap, los } from "../map.ts"
import type { Hello, Observation } from "../mechfight.ts"

const REPLAYS = Number(process.env.STRESS_REPLAYS || 5)
const FORCE = !process.env.STRESS_NO_FORCE // STRESS_NO_FORCE=1: normal cadence (diagnose)
const ARENA_DIR_DEFAULT = join(import.meta.dir, "../../kit/arena")
const dir = process.env.ARENA_DIR || ARENA_DIR_DEFAULT

const stream = join(import.meta.dir, "../../kit/check/stream.jsonl.gz")
const lines = gunzipSync(readFileSync(stream))
  .toString("utf8")
  .split("\n")
  .filter((l) => l.length > 0)
const msgs = lines.map((l) => JSON.parse(l) as { type: string } & Record<string, unknown>)
const hello = msgs.find((m) => m.type === "hello") as unknown as Hello
const obs = msgs.filter((m) => m.type === "obs") as unknown as Observation[]
if (!hello || obs.length === 0) {
  console.error(`stress: no hello/observations in ${stream}`)
  process.exit(1)
}

const map = loadMap(dir, hello.config.mech.step_up ?? 1.2)
const brain = new Brain(hello, map, (a, b) => los(map, a, b))
brain.act(obs[0], { elapsedMs: () => 0 }) // warm module paths once
// one untimed warmup replay: JIT + GC settle during it (a real match gives the bot a 20 s hello
// budget for the same purpose), so the MEASURED replays reflect steady-state act() latency
for (const o of obs) brain.act(o, { elapsedMs: () => 0, forceReplan: true })

const times: number[] = []
const slow: { rep: number; idx: number; ms: number }[] = []
let acted = 0
for (let r = 0; r < REPLAYS; r++) {
  for (const o of obs) {
    const t0 = performance.now()
    brain.act(o, { elapsedMs: () => performance.now() - t0, forceReplan: FORCE })
    const ms = performance.now() - t0
    times.push(ms)
    if (ms > 30) slow.push({ rep: r + 1, idx: times.length % obs.length, ms })
    acted++
  }
}

times.sort((a, b) => a - b)
const q = (p: number): number => times[Math.min(times.length - 1, Math.floor(p * times.length))]!
const mean = times.reduce((s, x) => s + x, 0) / times.length
const max = times[times.length - 1]!
console.log(`acts      : ${acted} (${REPLAYS} replays x ${obs.length} obs, ${FORCE ? "forced replan" : "normal cadence"}, capMs 10)`)
console.log(`p50       : ${q(0.5).toFixed(2)} ms`)
console.log(`p99       : ${q(0.99).toFixed(2)} ms`)
console.log(`mean      : ${mean.toFixed(2)} ms`)
console.log(`max       : ${max.toFixed(2)} ms`)
if (slow.length > 0) console.log(`slow(>30ms): ${slow.slice(0, 10).map((s) => `rep${s.rep}#${s.idx}=${s.ms.toFixed(0)}ms`).join(" ")}`)
console.log(max < 35 ? "verdict   : OK (max < 35 ms)" : "verdict   : REVIEW (max >= 35 ms)")
