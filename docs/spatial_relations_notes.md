# Spatial Relations Notes

## Current Synthetic Families

- `static_egocentric`: direct mapping from observer facing plus absolute offset
  to an egocentric relation.
- `absolute_role_swap`: same scene with figure and ground swapped, covering
  `north/south`, `east/west`, and `above/below`.
- `cardinal_pass_by`: an agent starts north/south/east/west of an object, then
  either walks toward the object and stops before reaching it or walks past it
  to the other side. Targets are balanced between the agent's relation to the
  object and the object's reciprocal relation to the agent.
- `cardinal_rotation_invariance`: an agent has an absolute cardinal relation
  to an object, then turns left/right/around. The target cardinal relation is
  unchanged, which separates absolute direction from body orientation.
- `cardinal_farther_motion`: an agent starts north/south/east/west of an
  object, then continues moving farther in that same direction. The target
  stays on the same cardinal side, with direct and role-swapped targets
  balanced.
- `vertical_paraphrase_role_swap`: vertical-only examples that balance literal
  swaps, paraphrase-to-literal mappings, and paraphrased role swaps for
  `above/below`, `over/under`, `on top of/beneath`, and
  `higher than/lower than`.
- `vertical_motion_update`: same-height scenes where either the figure or
  ground moves up/down, producing an `above/below` relation. Targets are
  balanced between direct statements and role-swapped inverse statements.
- `vertical_motion_distractor`: vertical motion updates with a second
  unrelated pair that can express the same or opposite relation and can appear
  before or after the target pair. This is meant to tighten entity binding.
- `egocentric_same_facing_role_swap`: two agents face the same direction, then
  the same scene is described from the other agent's role, covering
  `front/behind` and `left/right` without extra left/right sampling.
- `close_far_distance`: distance-only examples that treat `close/far` as
  symmetric scalar proximity rather than directional inversion. The first pass
  covers metric distance, movement toward/away, and symmetric role swap.
- `turn_update_basic`: the object stays fixed while the observer turns, so the
  relative relation changes.

The older `reciprocal_inversion` family name is still accepted as an alias for
absolute role swaps.

## Targeted Families To Consider Later

- Static egocentric grid: exhaustively balance every observer facing against
  every horizontal world offset.
- Facing contrast, same offset: keep the world relation fixed and change only
  the observer facing.
- Role-binding contrast: keep the text surface similar while swapping which
  entity is figure versus ground.
- Turn grid: balance every before-relation against left, right, and around
  turns.
- Motion transition: move one entity or both entities, then ask for the updated
  world or egocentric relation.
- Cardinal side language: map `north side/east side/...` to absolute cardinal
  relations.
- Distance/accessibility: handle `close/far` separately from directional
  templates, because the current auxiliary signal can regress that pair.

Keep left/right balanced against front/behind unless a probe explicitly needs a
left/right-only intervention.

## Prototype Mixes

- `spatial_relations_synth_context_rank_concept_balanced.yaml`: a separate
  5.5k context-rank prototype that balances broad concept pressure more
  explicitly. It uses 1.5k vertical rows, 1.5k cardinal rows, 1.5k egocentric
  rows, and 1k close/far rows. It excludes `absolute_role_swap` so cardinal and
  vertical exposure are easier to audit.

## Close/Far Subtypes

Minimal implemented subtypes:

- Metric distance: small step counts map to `close`; large step counts map to
  `far`.
- Movement toward/away: moving toward a referent makes the object close; moving
  away makes it far.
- Symmetric role swap: if an object is close/far to a person, the person is
  close/far to the object.

Useful future subtypes:

- Comparative distance: one object is closer/farther than another object.
- Path distance: barriers, walls, or routes can make straight-line proximity
  misleading.
- Reachability: reachable/unreachable depends on body limits, obstacles, and
  tools, not just distance.
- Same-area cues: same table, shelf, or room implies close; across the room,
  yard, hall, or street implies far.
- Perceptual access: in-hand, beside, nearby, or barely visible in the distance.
