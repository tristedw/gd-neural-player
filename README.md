# gd-neural-player

A small CNN that plays Geometry Dash Meltdown on macOS by looking at the game window and holding/releasing the mouse. It teaches itself a level by dying a lot.

No memory reading, no mods, no hooks. It only sees pixels and clicks, same as you.

## How it works

- **Capture**: finds the Meltdown window with Quartz and grabs it with `mss` (~70 fps on an M-series Air).
- **Input**: posts real mouse down/up events over the window. Holding = held tap, so ship, ball, UFO and wave all work the same way.
- **Vision**: frames get shrunk to 128x96, mixed with Canny edges (GD recolors levels constantly, edges stay consistent) and stacked 4 deep so the net can see motion.
- **Death detection**: phase correlation between frames measures how fast the level is scrolling. When the camera freezes the player died, when it moves again it respawned. No progress bar or text reading needed.
- **Net**: 4 conv layers + MLP, one output: probability of holding.
- **Learning** (`--learn`): every attempt is logged. On death, the ~0.5s before the crash gets relabeled to the opposite action (that's where the mistake was), everything before it is kept as good data, and attempts that beat the best distance get extra weight. A background thread retrains after each death and the new weights get swapped in on respawn. Exploration is cranked up near spots where it keeps dying.
- **Human demos** (optional): `record` logs you playing, which gives it a big head start.

## Setup

```bash
uv venv -p python3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

macOS permissions for whatever app you run it from (Terminal, iTerm, VS Code):

- **Screen Recording**: System Settings > Privacy & Security > Screen Recording. Without it you just get your wallpaper back.
- **Accessibility**: System Settings > Privacy & Security > Accessibility, so it can click.

Restart the terminal after granting. Check it works:

```bash
.venv/bin/python -m gdbot.probe --seconds 10
```

While you play you should see `dx` jump above 0 and `died` / `respawned` events.

## Usage

Open a level in Meltdown, leave the window visible and don't move it, then:

```bash
# let it learn the level from scratch (or keep improving the saved model)
.venv/bin/python -m gdbot.play --level seven_seas --learn

# optional: record yourself first for a head start, then train on it
.venv/bin/python -m gdbot.record --level seven_seas
.venv/bin/python -m gdbot.train --level seven_seas --epochs 10

# just play with the trained model, no learning
.venv/bin/python -m gdbot.play --level seven_seas
```

Ctrl+C stops it and saves `models/<level>.pt`. Data lives in `data/<level>/`.

Practice mode is a good idea for long sessions since the run restarts at checkpoints, but normal mode works too.

Flags worth knowing:

| flag | default | what |
| --- | --- | --- |
| `--explore` | 0.02 | base chance per frame to try the opposite input |
| `--threshold` | 0.5 | hold if p(hold) is above this |
| `--fps` | 60 | loop rate cap |

## Tests

```bash
.venv/bin/python -m tests.test_offline
```

Runs the death detector against a fake scrolling level and does a training pass. Doesn't need the game.

## Notes

- Meltdown pauses itself when it loses focus. `play` brings it to the front at start, keep it there.
- The free version shows ads now and then. The bot can't close those, if one pops up close it and it'll pick back up when the level scrolls again.
- One model per level works best. You can train on several levels into one model by pointing `--level` at the same name.
- Retina displays: capture uses logical points so it doesn't matter.
