# AeroMind AI — Drone Operations

## Overview
AeroMind AI is a drone intelligence system integrating
PX4 Autopilot, Gazebo simulation, MAVSDK, computer vision,
and local artificial intelligence.

## Simulation
AeroMind AI uses PX4 SITL with Gazebo to test drone
operations in a simulated environment.

## Flight Operations
The flight controller manages:
- Vehicle connection
- Telemetry monitoring
- Preflight readiness checks
- Arming
- Takeoff
- Hovering
- Landing

## AI Integration
Llama 3.2 provides advisory intelligence through Ollama.

The RAG engine retrieves relevant information from
indexed knowledge documents.

## Safety
AI-generated responses must not directly execute
flight commands or override independent safety checks.
