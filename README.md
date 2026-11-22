# Mech Royale bot

This repository contains the starter TypeScript bot for the Builders Buildathon's Mech Royale track: a 16-mech free-for-all battle royale where each bot pilots a mech on the Ohio State campus by reading JSON observations on stdin and writing JSON actions on stdout (spec: `mechfight/1`). It is built as a Docker image from the pinned `oven/bun:1.4.4-slim` base and run sandboxed (no network, 1 CPU, 512 MB) by the event's server.

## Experimental branch: TTUN/r1ver economy bot (NOT our competition entry)

Experimental branch: TTUN/r1ver economy bot, MIT, Copyright (c) 2026 TTUN, via Evans-Kwashie/mech-royale-bot baselines/r1ver @ af337d5. Not our competition entry.

Baseline capture note: the pinned upstream revision af337d5c1275d03478685bd4936c8a551fa6b200 is not reachable in the upstream repository (its git history was rewritten; the full clone holds 41 commits, none matching that SHA). The bot.ts / map.ts / mechfight.ts here are byte-identical copies from 3fd07bebc667741893920a0286f1e5f2b8a01420 (2026-11-21, "Improve Mech Royale economy and shared weapon control") — the only revision of baselines/r1ver ever published upstream. Its license is kept verbatim as LICENSE-TTUN.
Fidelity note: one shim added (mirroring our own bot.ts) — the Windows-host conformance harness abspath()es "/arena" to "C:\arena" in hello.map.dir; the constructor maps that back to /arena so the map actually loads in the sandbox. Without it the bot runs map-blind in Docker. Nothing else changed.
