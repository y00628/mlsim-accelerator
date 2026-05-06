#!/usr/bin/env python3
"""
Parse and summarize cycle stats logged by the SystemC simulation.
Fill in during Phase 2 (June) once tb_array.cpp emits stats.

Expected log format (one line per simulation run):
  STATS array_size=8 M=64 K=64 N=64 cycles=1234 util=0.87 stalls=56

Outputs:
  - Utilization vs array size table
  - Stall breakdown pie chart
  - Throughput (ops/cycle) vs matrix size
"""
