# BDQN Mission Belief20 UAV

Single-drone UAV search/track environment combining:

- **BDQN mission decision** : the learned agent chooses only `SEARCH` or `TRACK`.
- **20x20 belief-memory maps** : the drone only senses locally, then updates its own 20x20 memory map.

## Key idea

The real world is hidden and full-size:

```text
truth environment: 20 x 20
hidden target positions: 20 x 20
```

The drone sensor is local:

```text
sensor radius = 2 by default
```

But the drone maintains its own internal memory over the full map:

```text
belief_map          20 x 20
visited_map         20 x 20
known_target_map    20 x 20
completed_map       20 x 20
```

So the drone does **not** see the global truth. It only updates a global-sized memory from local observations.

## Who uses which map?

### Environment

The environment owns the hidden truth:

```text
target_pos
true target values
completion flags
```

This is used only for simulation, detection, reward, and evaluation.

### BDQN

BDQN receives the drone memory state:

```text
obs shape = (5, 20, 20)
```

Channels:

```text
0. drone position map
1. belief map from drone memory
2. known target value map
3. completed target map
4. visited map
```

BDQN outputs:

```text
0 = SEARCH
1 = TRACK
```

### Controller

The low-level controller receives the same memory, not the hidden truth.

If mission is `SEARCH`, it moves toward high belief / high uncertainty cells.

If mission is `TRACK`, it moves toward the best known non-completed target. If no target is known, it falls back to SEARCH with a small penalty.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Smoke test

```bash
python scripts/smoke_test.py
```

## Train

```bash
python scripts/train.py --episodes 500 --macro-steps 5 --sensor-radius 2
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/latest.pt --episodes 100 --macro-steps 5 --sensor-radius 2
```

## Why this is the right base for QMIX

Later, in QMIX:

```text
Each UAV has its own 20x20 memory map.
QMIX assigns SEARCH/TRACK to each UAV.
Each low-level controller uses only its own UAV memory.
The mixer can use the global state during training only.
```
