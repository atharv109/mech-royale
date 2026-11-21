# Mech Royale bot

This repository contains the starter TypeScript bot for the Builders Buildathon's Mech Royale track: a 16-mech free-for-all battle royale where each bot pilots a mech on the Ohio State campus by reading JSON observations on stdin and writing JSON actions on stdout (spec: `mechfight/1`). It is built as a Docker image from the pinned `oven/bun:1.4.4-slim` base and run sandboxed (no network, 1 CPU, 512 MB) by the event's server.
