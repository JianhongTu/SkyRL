# Multi-Env RL

This repo contains the codebase for the SWE phase of the multi-env rl project. 

## Setup

We will train on willhx/Qwen3-30B-A3B_base_math_search. 

## SFT

Candidate dataset is nvidia/Nemotron-SFT-SWE-v3. However, it aggregates trajectories from multiple agent harnesses, including OpenHands, SWE-agent and MSA. We filter out trajectories not collected from OpenHands. For the remaining trajectories, we modify the tool call format by striping away unexpected arguments including security_risk, timeout, etc. Those using un unsupported tool (task_trakcer) are thrown away. 
