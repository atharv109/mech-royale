/**
 * sim/botio.ts — runs an external bot as a mechfight/1 subprocess: hello/ready handshake,
 * one obs per decision with the 50 ms budget, stale-answer skipping, coast/idle per §2.
 */
import { type Hello } from "./config.ts"

export type SubBot = {
  proc: Bun.Subprocess
  reader: ReadableStreamDefaultReader<Uint8Array>
  buf: string
  name: string
  lastAction: Record<string, unknown>
  lastAnswerT: number // match time of last on-time answer
  exited: boolean
  stderrText: string
}

const enc = new TextEncoder()
const dec = new TextDecoder()

export async function spawnBot(cmd: string[], cwd: string | undefined, hello: Hello, env?: Record<string, string>): Promise<SubBot> {
  const proc = Bun.spawn({
    cmd,
    cwd,
    env: { ...process.env, ...env } as Record<string, string | undefined>,
    stdin: "pipe",
    stdout: "pipe",
    stderr: "piped",
  })
  const reader = proc.stdout.getReader() as ReadableStreamDefaultReader<Uint8Array>
  const bot: SubBot = { proc,  buf: "", name: "bot", lastAction: {}, lastAnswerT: 0, exited: false, stderrText: "" }
  // continuously drain stderr so it can never block the bot (and keep the last BOTSIG line)
  {
    const sr = (proc.stderr as unknown as ReadableStream<Uint8Array>).getReader()
    void (async () => {
      for (;;) {
        try {
          const { done, value } = await sr.read()
          if (done) break
          bot.stderrText += dec.decode(value)
          if (bot.stderrText.length > 200000) bot.stderrText = bot.stderrText.slice(-100000)
        } catch { break }
      }
    })()
  }
  bot.reader = reader
  bot.proc.stdin!.write(enc.encode(JSON.stringify(hello) + "\n"))
  await bot.proc.stdin!.flush()
  // ready within hello budget
  const ready = await readLine(bot, hello.config.bot.hello_budget_ms)
  if (!ready) throw new Error(`bot did not send ready: ${cmd.join(" ")}`)
  try {
    const r = JSON.parse(ready.trim())
    bot.name = String(r.name ?? "bot").slice(0, 32)
  } catch {
    /* unnamed */
  }
  return bot
}

export async function readLine(b: SubBot, timeoutMs: number): Promise<string | null> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    const idx = b.buf.indexOf("\n")
    if (idx >= 0) {
      const line = b.buf.slice(0, idx)
      b.buf = b.buf.slice(idx + 1)
      return line
    }
    const left = deadline - Date.now()
    if (left <= 0) return null
    const timer = new Promise<null>((res) => setTimeout(() => res(null), left))
    const chunk = await Promise.race([b.reader.read(), timer])
    if (chunk === null) continue
    if ((chunk as { done?: boolean }).done) {
      b.exited = true
      return null
    }
    b.buf += dec.decode((chunk as { value: Uint8Array }).value)
  }
}

/** One decision: send obs, wait within budget for the newest answer; null on miss. */
export async function decide(b: SubBot, obs: Record<string, unknown>, budgetMs: number): Promise<Record<string, unknown> | null> {
  b.proc.stdin!.write(enc.encode(JSON.stringify(obs) + "\n"))
  await b.proc.stdin!.flush()
  const deadline = Date.now() + budgetMs
  for (;;) {
    const line = await readLine(b, Math.max(1, deadline - Date.now()))
    if (line === null) return null
    let a: Record<string, unknown> | null = null
    try {
      a = JSON.parse(line.trim())
    } catch {
      continue // bad line: ignored
    }
    if (a && a.tick === obs.tick) return a
    // stale or missing tick: drop, keep listening for the newest
    if (Date.now() >= deadline) return null
  }
}

/** Coast transform: keep continuous parts, drop one-shots. */
export function coastOf(last: Record<string, unknown> | null): Record<string, unknown> {
  if (!last) return {}
  return { move: last.move, boost: last.boost, aim: last.aim }
}

/** Drain stderr after stdin end and parse the last "BOTSIG {...}" line (bot-side counters). */
export function botsigOf(b: SubBot): Record<string, unknown> | null {
  for (const line of b.stderrText.split("\n")) {
    if (line.startsWith("BOTSIG ")) {
      try { return JSON.parse(line.slice(7).trim()) as Record<string, unknown> } catch { return null }
    }
  }
  return null
}

export async function endBot(b: SubBot, place: number, kills: number): Promise<Record<string, unknown> | null> {
  try {
    b.proc.stdin!.write(enc.encode(JSON.stringify({ type: "end", place, kills }) + "\n"))
    await b.proc.stdin!.flush()
    await b.proc.stdin!.end()
  } catch {
    /* already gone */
  }
  const t = setTimeout(() => {
    if (!b.exited) try { b.proc.kill() } catch {}
  }, 2000)
  void t
  // give the exit hook a beat to flush BOTSIG onto stderr
  const dl = Date.now() + 700
  while (Date.now() < dl && !b.exited) await new Promise((r) => setTimeout(r, 50))
  return botsigOf(b)
}
