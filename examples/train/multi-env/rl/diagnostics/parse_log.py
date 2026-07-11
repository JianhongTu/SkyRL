"""Diagnostic: summarize a rollout log (stdout of collect_batch.py, redirected to
a file) into the symptoms in ../README.md — end-reason, turn length, finish rate,
format pathologies, and the blocking-command hang count. Pure regex over the log;
no model or extra deps.

Usage:
    python parse_log.py <rollout.log> [max_iterations]

    <rollout.log>    the captured stdout of a collect_batch.py run
    max_iterations   turn cap used for that run (default 50); trajectories that
                     reach it are the ones the trainer masks out
"""
import re, json, sys, statistics as st
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    sys.exit("usage: python parse_log.py <rollout.log> [max_iterations]")
LOG = sys.argv[1]
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 50
raw = open(LOG, errors="replace").read()
print(f"log chars: {len(raw):,}  (cap={CAP})")

# ---- per-turn response blocks: "instance id X, trajectory T, response ... stop reason (stop|length)"
TURN = re.compile(r"instance id (\S+?), trajectory (\d+), response (.*?) stop reason (stop|length)\b", re.S)
turns = defaultdict(list)
for m in TURN.finditer(raw):
    turns[(m.group(1).rstrip(","), m.group(2))].append({"resp": m.group(3), "stop": m.group(4)})

STEP = re.compile(r"instance id (\S+?), trajectory (\d+), step (\d+)")
maxstep = defaultdict(int)
for m in STEP.finditer(raw):
    k = (m.group(1).rstrip(","), m.group(2)); maxstep[k] = max(maxstep[k], int(m.group(3)))

ntraj = len(turns)
all_turns = [t for v in turns.values() for t in v]
print(f"trajectories parsed: {ntraj}  total turns: {len(all_turns)}")
if not all_turns:
    sys.exit("no turn records found (is this a collect_batch.py log?)")

def tool_names(s):
    names = []
    for mm in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", s, re.S):
        try: names.append(json.loads(mm.group(1)).get("name"))
        except Exception: names.append("<unparse>")
    return names

lens = sorted(len(t["resp"]) for t in all_turns)
length_turns = [t for t in all_turns if t["stop"] == "length"]
fab = [t for t in all_turns if ("<response>" in t["resp"] or "<tool_response>" in t["resp"])]
func = [t for t in all_turns if "<function=" in t["resp"]]
multi = [t for t in all_turns if t["resp"].count("<tool_call>") > 1]
notc = [t for t in all_turns if "<tool_call>" not in t["resp"]]

print("\n## Q4 TURN LENGTH (chars; ~tokens = chars/3.5)")
print(f"  mean={st.mean(lens):.0f} median={st.median(lens)} p90={lens[int(.9*len(lens))]} "
      f"p99={lens[int(.99*len(lens))]} max={max(lens)}")
print(f"  turns >8k chars: {sum(x>8000 for x in lens)}  >40k: {sum(x>40000 for x in lens)}")

print("\n## Q3 3-PHASE-RELEVANT TURN PATHOLOGIES")
print(f"  stop=length (runaway/cap):   {len(length_turns)}/{len(all_turns)} ({100*len(length_turns)/len(all_turns):.2f}%)")
print(f"  fabricated <response>:       {len(fab)}")
print(f"  <function=> drift:           {len(func)}")
print(f"  >1 <tool_call> per turn:     {len(multi)}")
print(f"  no <tool_call> at all:       {len(notc)}")
benefit = {k for k, v in turns.items()
           if any(t["stop"] == "length" or "<response>" in t["resp"]
                  or t["resp"].count("<tool_call>") > 1 or "<function=" in t["resp"] for t in v)}
print(f"  => trajectories 3-phase would touch: {len(benefit)}/{ntraj}")

print("\n## Q2 END REASON")
finish_traj = {k for k, v in turns.items() if any("finish" in tool_names(t["resp"]) for t in v)}
capped = {k for k in turns if maxstep.get(k, 0) >= CAP}
print(f"  reached cap (max_iterations -> MASKED OUT): {len(capped)}/{ntraj} ({100*len(capped)/ntraj:.0f}%)")
print(f"  ever emit a finish tool_call:               {len(finish_traj)}/{ntraj}")
steps = sorted(maxstep.values())
print(f"  max-step per traj: median={st.median(steps)} min={min(steps)} <cap={sum(s<CAP for s in steps)}")

print("\n## Q4 NEW PATTERN — blocking-command hang")
print(f"  'previous command is still running' observations: "
      f"{raw.count('is NOT executed. The previous command is still running')}")

allnames = Counter()
for t in all_turns: allnames.update(tool_names(t["resp"]))
print("\n## tool-call name distribution:", dict(allnames.most_common()))

print("\n## sample stop=length turn tails:")
for t in length_turns[:4]:
    print("  --", repr(t["resp"][-260:]))
