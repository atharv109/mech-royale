# Build from this folder: docker build -t my-bot .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
# JIT off (interpreter mode): on the 1-CPU quota, bun's background JIT/GC threads steal the main
# core and cause ~50 ms stop-the-world stalls; interpreter mode keeps p50 ~0.7 ms, max ~12 ms
# (stress: 5x300 acts, forced replan, 2026-11-20). The A* cap is wall-clock, so slower bytecode
# only shortens partial paths, never blows the deadline.
FROM oven/bun:1.4.4-slim
ENV BUN_JSC_useJIT=0
WORKDIR /bot
COPY . .
ENTRYPOINT ["bun", "/bot/bot.ts"]
