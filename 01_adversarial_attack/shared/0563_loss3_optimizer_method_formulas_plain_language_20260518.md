# Loss3 Optimizer Methods, Plain-Language Formula Version - 2026-05-18

This version avoids LaTeX source blocks and code-style formulas. It writes the
optimizer updates in plain readable text.

## Symbols

delta means the perturbation.

loss means the current attack objective:

loss(delta) means the q-norm of model output minus solver output after adding delta to the input.

In words:

loss(delta) = size of [model(x + delta) - solver(x + delta)], measured with q-norm.

At step k:

current perturbation = delta_k

current gradient = g_k

That gradient means:

g_k = the gradient of loss with respect to delta, evaluated at delta_k.

The perturbation is not allowed to leave the p-norm ball:

size_p(delta) must be no bigger than epsilon.

Projection means:

If the proposed delta is inside the allowed ball, keep it.

If it is outside the allowed ball, shrink it back to the boundary.

For p = infinity, projection means clipping every coordinate into [-epsilon, epsilon].

## Two Direction Types

Raw gradient direction:

direction = g_k

Normalized raw-gradient direction:

direction = g_k divided by its p-norm.

Lp-steepest direction:

direction = the p-norm unit direction that gives the largest increase of the loss under the current gradient.

Important special case:

When p = 2, the Lp-steepest direction is exactly the normalized gradient.

So for p = 2:

Lp-steepest direction = normalized raw-gradient direction.

This is the main reason several p=2 methods are identical.

## Two Update Types

Additive update:

new delta = old delta + alpha times direction, then project back to the allowed ball.

Replacement update:

new delta = epsilon times direction, then project back to the allowed ball.

Replacement update throws away the previous delta and jumps directly to the boundary direction.

## Method Formulas In Plain Words

### raw_add

Direction used: raw gradient.

Update:

new delta = old delta + alpha times raw gradient, then project.

Formula in plain text:

delta_next = project(delta_current + alpha * g)

This is ordinary PGD.

### unit_raw_add

Direction used: normalized raw gradient.

Update:

new delta = old delta + alpha times normalized raw gradient, then project.

Formula in plain text:

delta_next = project(delta_current + alpha * normalized_gradient)

### raw_replace

Direction used: normalized raw gradient.

Update:

new delta = epsilon times normalized raw gradient, then project.

Formula in plain text:

delta_next = project(epsilon * normalized_gradient)

This jumps straight to the boundary in the normalized-gradient direction.

### steepest_add

Direction used: Lp-steepest direction from the gradient.

Update:

new delta = old delta + alpha times Lp-steepest direction, then project.

Formula in plain text:

delta_next = project(delta_current + alpha * steepest_direction)

This is LP-steepest PGD.

### steepest_replace

Direction used: Lp-steepest direction from the gradient.

Update:

new delta = epsilon times Lp-steepest direction, then project.

Formula in plain text:

delta_next = project(epsilon * steepest_direction)

This is the earlier generalized-power-style method from the three-method comparison.

### power_add__objective_gradient

Direction used: Lp-steepest direction from the objective gradient.

Update:

new delta = old delta + alpha times Lp-steepest direction, then project.

So this is the same formula as steepest_add.

### power_replace__objective_gradient

Direction used: Lp-steepest direction from the objective gradient.

Update:

new delta = epsilon times Lp-steepest direction, then project.

So this is the same formula as steepest_replace.

## Why Some Similarities Are Exactly 1

In the p=2, q=2 run, the first seven methods are:

1. raw_add
2. unit_raw_add
3. raw_replace
4. steepest_add
5. steepest_replace
6. power_add__objective_gradient
7. power_replace__objective_gradient

These seven all use the ordinary objective gradient. They are not JVP/VJP methods.

For p = 2:

normalized raw gradient = L2-steepest direction.

Therefore:

unit_raw_add is the same method as steepest_add.

steepest_add is the same method as power_add__objective_gradient.

So:

unit_raw_add = steepest_add = power_add__objective_gradient.

Also:

raw_replace is the same method as steepest_replace.

steepest_replace is the same method as power_replace__objective_gradient.

So:

raw_replace = steepest_replace = power_replace__objective_gradient.

That is why their final perturbation similarity is exactly 1.

It is not a mysterious result. It is formula duplication.

## Why raw_add Is Similar But Not Identical

raw_add uses the raw gradient itself.

steepest_add uses the normalized gradient when p = 2.

So they point in similar directions, but they do not take exactly the same step sizes.

That is why raw_add vs steepest_add has high similarity, but not exactly 1.

Observed p=2,q=2 result:

raw_add versus steepest_add:

mean cosine similarity = 0.858049

mean relative L2 distance = 0.384210

## Why additive and replacement are not the same

steepest_add keeps the old delta and adds a new step.

steepest_replace throws away the old delta and directly jumps to epsilon times the current direction.

So even if they use the same direction type, the update rule is different.

Observed p=2,q=2 result:

steepest_add versus steepest_replace:

mean cosine similarity = 0.530025

mean relative L2 distance = 0.830928

## JVP/VJP Power Methods In Plain Words

These are different from the first seven methods.

They do not directly use the ordinary objective gradient as the direction source.

Instead, they use the local Jacobian of the residual map.

The residual map means:

residual(input) = model(input) - solver(input).

JVP means:

Take the current power direction and push it through the residual Jacobian.

VJP means:

Take the q-side output direction and pull it back through the transpose Jacobian.

Then the method converts that pulled-back vector into an Lp-steepest direction.

In plain words:

power JVP/VJP direction = first push a direction through the model-solver residual, then pull it back, then choose the p-steepest perturbation direction.

Additive JVP/VJP update:

new delta = old delta + alpha times that JVP/VJP direction, then project.

Replacement JVP/VJP update:

new delta = epsilon times that JVP/VJP direction, then project.

Current generalized_pq is implemented the same as pure_jvp_vjp in this script, so those two are currently redundant.

## Main takeaway

The p=2,q=2 first-seven similarity is mainly because several methods are actually the same formula.

The real independent distinctions are:

raw gradient versus normalized or steepest gradient;

additive update versus replacement update;

objective-gradient direction versus JVP/VJP operator-power direction;

p=2 geometry versus p=1 or p=inf geometry.
