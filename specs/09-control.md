# 09 — Control Arm

A second, *non-evolving* arm that runs in parallel with the evolving arm
every iteration. Same actor, same judge, same seed prompt forever. Its
purpose is purely **epistemic**: it gives the audience a visual against
cherry-picking.

## Spec

- Uses the **identical** seed prompt every iteration:
  `"You are an ASCII artist. Draw the requested subject."`
- Generates `k=8` candidates with the same temperature / thinking level
  as the evolving arm.
- Renders + judges identically. Same `JudgeResult` schema.
- **Does not call the rewriter.** Its prompt is frozen.
- Recorded in `RunHistory` as `arm="control"` records, interleaved with
  evolving records.

## What we expect

- Evolving win-rate curve: starts ~0.2, climbs to ~0.6 over 6 iterations.
- Control win-rate curve: hovers ~0.2 with random noise. No trend.

If the control arm *also* climbs, that's a tell — either the judge is
biased toward later iterations (memory effect) or the seed prompt is
already strong enough that the actor randomly stumbles into better outputs.
Either way, the demo loses its punch. Watch this in rehearsal.

## Toggle

`--no-control` on the CLI for cost-saving rehearsal runs. The UI hides
the second line when no control data exists.

## Test surface

Most of the control-arm logic is just calling actor/renderer/judge with a
fixed prompt — covered by `loop_test.py`. A targeted test verifies that
the control prompt at iteration N is byte-identical to the seed prompt.
