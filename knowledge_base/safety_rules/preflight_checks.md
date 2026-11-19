# AeroMind AI — Preflight Safety Procedures

## 1. Flight Controller Connection
Before takeoff, AeroMind AI must verify that the
PX4 flight controller is connected successfully
through MAVSDK.

## 2. Position and Navigation Health
The system must verify that the position estimation
and navigation health requirements for the intended
flight mode have been satisfied.

## 3. Battery Verification
Battery telemetry must be available and the reported
battery level must satisfy configured safety limits.

## 4. Simulation Environment
For simulated missions, PX4 SITL and Gazebo must
be running correctly.

## 5. Vehicle Readiness
The vehicle must satisfy the required arming checks
before a takeoff command is permitted.

## 6. Mission Safety
The mission must be checked against configured
operational restrictions and safety requirements.

## 7. Operator Authorisation
Test missions requiring operator approval must
receive that approval before execution.

## 8. Emergency Procedures
The system must retain its configured failsafe
mechanisms and appropriate landing procedures.

## 9. RAG Safety Restrictions
Retrieved knowledge is advisory only. The language
model must never directly authorise flight commands
or bypass independent safety validation.

