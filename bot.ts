/** Mech Royale bot v1: wires the map, brain, and a last-resort fallback. Bot behavior lives in brain.ts. */
import { log, run, type BotAction, type Hello, type Observation } from "./mechfight.ts"
import { loadMap, los, type GameMap } from "./map.ts"
import { Brain } from "./brain.ts"

/** Load the arena. On a Windows host the conformance harness abspath()es "/arena" to "C:\arena";
 * map that back to the in-container mount before giving up (Linux hosts pass "/arena" through). */
function initMap(hello: Hello): GameMap {
  const stepUp = hello.config.mech.step_up ?? 1.2
  try {
    return loadMap(hello.map.dir, stepUp)
  } catch (e) {
    const remap = hello.map.dir.replace(/^[A-Za-z]:[\\/]/, "/")
    if (remap === hello.map.dir) throw e
    return loadMap(remap, stepUp)
  }
}

run((hello) => {
  let brain: Brain | null = null
  try {
    const map: GameMap = initMap(hello)
    brain = new Brain(hello, map, (a, b) => los(map, a, b))
  } catch (e) {
    // stdout is the protocol — everything we can say goes to stderr
    log("mech-royale: brain init failed, running fallback:", e)
  }

  const fallback = (obs: Observation): BotAction => {
    // spec v1 fallback: move toward zone center, aim nearest visible mech, fire the gun
    const me = obs.self
    const [x, y, z] = me.pos
    const c = obs.zone.center
    const yaw = Math.atan2(c[1] - y, c[0] - x)
    const enemy = obs.mechs
      .filter((m) => m.kind === "mech")
      .sort((a, b) => (a.pos[0] - x) ** 2 + (a.pos[1] - y) ** 2 - ((b.pos[0] - x) ** 2 + (b.pos[1] - y) ** 2))[0]
    let aim: [number, number] = [yaw, 0]
    if (enemy) {
      const d = Math.hypot(enemy.pos[0] - x, enemy.pos[1] - y)
      aim = [Math.atan2(enemy.pos[1] - y, enemy.pos[0] - x), Math.atan2(enemy.pos[2] + 4.5 - z - 4.5, d)]
    }
    return { move: [Math.cos(yaw), Math.sin(yaw)], aim, fire: [enemy != null, false] }
  }

  // Task 10: behavior counters emitted as "BOTSIG {...}" on stderr at exit (sim captures it)
  const stats: Record<string, unknown> = brain
    ? brain.stats
    : { dodges: 0, retreats: 0, coverTicks: 0, seen: {}, seenDist: {}, walked: 0, took: {}, fail: {} }
  let maxActMs = 0
  process.on("exit", () => {
    try {
      console.error("BOTSIG " + JSON.stringify({ ...stats, maxActMs: Math.round(maxActMs * 100) / 100 }))
    } catch {}
  })

  return {
    name: "mech-royale-v1",
    act(obs: Observation): BotAction {
      // hard deadline: A* inside brain.act runs only while act is still under 10 ms of wall
      // time (cap dropped to 10 ms); past that the brain reuses its last path
      const t0 = performance.now()
      const done = (r: BotAction): BotAction => {
        const ms = performance.now() - t0
        if (ms > maxActMs) maxActMs = ms
        return r
      }
      if (brain) {
        try {
          return done(brain.act(obs, { elapsedMs: () => performance.now() - t0 }))
        } catch (e) {
          log(`brain.act failed at tick ${obs.tick}:`, e)
          brain = null // don't rethrow every tick: fall back permanently
        }
      }
      return done(fallback(obs))
    },
  }
})
