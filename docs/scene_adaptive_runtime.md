# Scene-adaptive runtime and 500 ms deployment policy

## Zero and weak-height research refinement (V32, 2026-10-12)

See [V32 paired independent evaluation](transparent_focus_v32.md). The same
five prior capture groups and native-image matcher are retained; reference-
specific monotone cubic calibration and soft robust fusion replace the global
linear latent response. Actual height remains 0--9 m and camera-to-bottom B
at most 10 m. No zero snapping, output clipping or confidence/coverage gating
is introduced in the research evaluation.

On 8016 new groups, zero-height MAE is 5.79 to 4.92 mm (52.95% to 62.67% passes),
0<H<1 m is 11.17 to 5.74 mm (50.36% to 75.14%), and 2<=H<3 m is 44.49 to
7.67 mm (57.14% to 100%). The 3504-group uniform subset passes 97.20% versus
87.99%, with 10.26 versus 23.35 mm MAE. Zero and very shallow translucent cases
remain weak; the zero MAE improvement cluster interval crosses zero. Maximum
error increases slightly to 284.67 mm despite improved means and P95.

Single-thread uncached runtime checks preserve predictions exactly and are
approximately 0.23--0.27 s P95, excluding acquisition, transfer and display.
They do not certify 500 ms end-to-end timing. Numeric output is not qualified
coverage; production defaults, safety gates and earlier evidence are unchanged.

## Actual liquid height range (V31, 2026-10-12)

The requested actual bottom-relative liquid height H spans 0--9 m, while the
camera-to-bottom distance B remains at most 10 m. The two ranges are not
interchangeable. See [V31 all-frame range validation](transparent_range_v31.md)
for the frozen five-capture research calibration, independent uniform-height
test, one-metre strata and uncached processing timing. Zero height has no
defined AbsRel and is assessed separately in millimetres. Small containers
below 96 native empty-bottom pixels are no longer an optimization target.

Deep-path attenuation can remove the bottom evidence required by this route;
such material is audited separately, not silently assigned to the ordinary
surface route. Numeric diagnostic output is not qualified coverage. The
research matcher is not promoted to production, and existing default routes,
safety gates and the 500 ms end-to-end deployment target remain unchanged.

## Finite-bottom small-image validation (V30, 2026-10-11)

See [V30 finite-bottom all-frame test](transparent_finite_v30.md). With five prior
capture groups, the frozen research recipe produces numeric values for all 1728
new test groups and passes 65.22%, but MAE increases from the direct-registration
physical-model control's 19.68 mm to 67.15 mm. Weak responses below 32 pixels cause
severe inverse-calibration tails. The candidate is not promoted to production.
At least 96 pixels: 1.38 mm MAE and 96.01% passes; below 32 pixels: 290.17 mm and
8.33%. These strata do not define a newly validated production routing threshold.

`FiniteBottomRelativeMeter` reproduces the frozen research recipe using fixed
empty-bottom ROIs and five rectified stereo prior groups. It requires unchanged
camera/container/texture and known geometry; automatic pose-change detection is
not provided. A six-frame, 18.75-pixel uncached processing check matches frozen
predictions exactly and measures 58.09 ms P95 on one CPU thread, excluding prior
fitting, acquisition, transfer and display. It does not certify end-to-end timing.
Production defaults and safety gates remain unchanged.

## Empty plus four relative-image priors (V29, 2026-10-11)

See [V29 independent all-frame test](transparent_relative_v29.md). The frozen
development policy uses one empty and four known liquid heights, with multi-prior
same-channel bottom-texture tracking and bounded residual calibration. On 864 new
stereo transparent/mild-translucent samples, MAE falls from 6.674 to 0.691 mm;
max(5 mm, 2%H) passes rise from 546/864 to 864/864. Strict per-frame 1% passes are
621/864, not 100%. These are unfiltered research measurements, not qualified
production coverage. The original baseline is rerun on this same new test set.

`FixedCameraRelativeMeter` is a separate resident research interface using five
observed stereo image groups, known liquid heights and calibrated geometry. It
requires a fixed camera/container/bottom texture, visible bottom and common
rectified intrinsics under the top-view flat geometry assumption. Parallel
matching does not change the frozen core. Timing smoke tests do not certify
camera-to-display 500 ms. Infinite-field texture and H/B scattering proxies do
not validate small finite vessels, pose changes, strong scattering or hardware.
Production defaults and safety gates remain unchanged.

## Capacity-limited five-prior validation (V28, 2026-10-11)

See [V28 independent all-frame validation](transparent_capacity_v28.md). All five
known liquid heights stay below the feasible container maximum C, independently
of camera-to-bottom distance B. The frozen development-selected layout yields
4.085 mm MAE and 585/720 passes versus 4.142 mm and 587/720 for the capacity-uniform
baseline; paired improvement intervals cross zero. No layout promotion is made.
All frames remain in the pass denominator, without confidence/coverage filtering
or clipping numeric estimates to capacity. Oracle correspondences diagnose
matching/calibration coupling only and are not deployable measurements. This
flat-bottom, infinite-field simulation does not qualify finite real vessels,
material scattering, production accuracy, or 500 ms end-to-end timing. Production
defaults and safety gates remain unchanged.

## Five-level shallow-liquid calibration research (V27, 2026-10-11)

See [V27 all-frame independent test](transparent_shallow_v27.md). Four five-frame
known-liquid-height layouts are compared on development data, followed by residual
pixel/phase-bias candidates. Policy and source are frozen before independent test
generation. Diagnostic estimates remain unfiltered and unqualified; production
defaults are unchanged. Simulated high-level priors may be infeasible in a shallow
real vessel, so capacity-constrained calibration requires separate validation.

## Homogeneous-refraction research update on 2026-10-11 (V26)

See [V26 all-frame comparison](transparent_refinement_v26.md). Bounded empirical
residual correction, stereo subpixel-bias calibration, and structured-light block
aggregation are implemented as an independent research route. Neither confidence
nor coverage filtering is used in these diagnostic tests. V26 does not replace
production defaults or certify industrial output. The 30-exposure structured-light
measurement requires acquisition timing in addition to its processing benchmark;
it is not currently qualified for the 500 ms end-to-end policy.

## Explicit route boundaries added on 2026-10-11 (V25)

See [scene route boundaries](scene_route_boundaries_v25.md). New opt-in
`measurement_route` separates verified surface returns, homogeneous transparent
refraction, and multilayer research. Strong-scattering surface candidates are
excluded from transparent optimization; a surface return must be independently
verified before using the standard surface route. Old profiles remain
`legacy_unspecified`; the old RGB/depth trigger is not a material classifier.
V25's unfiltered diagnostic estimates are research outputs, not production
confidence-qualified measurements. Production safety gates remain in force.

## Liquid-height qualification correction on 2026-10-11

Historical V6/V8 confidence qualification used an axial camera-depth proxy, not
bottom-referenced liquid height. Those reports cannot qualify product liquid-depth
routes. Specialists remain disabled. New metric definitions and per-scene results
are in [Liquid height reassessment](liquid_height_reassessment_v21.md). V17/V20
research entries are not automatically connected to the default industrial panel.

## Outcome

The runtime has three operator-selectable policies:

- `off`: always use the standard model;
- `auto`: inspect the current container ROI and request a specialist only when needed;
- `always`: force the operator-selected specialist for a qualified production profile.

The industrial panel exposes the same three choices. Video streams use an eight-frame
hold after an automatic trigger, avoiding model oscillation near a threshold. A frame-local
`scene_context.json` overrides the profile and can force a model on or off.

## Automatic decision

The default auto policy requests a specialist when any of these conditions is observed:

- valid raw depth inside the liquid mask/ROI falls below 45%;
- saturated highlights exceed 10%;
- median luminance falls below 0.18;
- dark pixels exceed 70%;
- luminance P90-P10 dynamic range falls below 0.06;
- the operator declares a transparent vessel/liquid, multi-layer ray, glare, or low-light scene.

Variant priority is transparent/multi-layer, glare, low light, then generic depth failure.
An explicit `force_complex_model: false` always wins over hysteresis.

Example frame context:

```json
{
  "transparent_container": true,
  "transparent_liquid": true,
  "multi_layer_expected": true,
  "glare_expected": false,
  "force_complex_model": null,
  "model_variant": "transparent_multilayer"
}
```

Place this file next to `rgb.png`, `depth.npy`, and `depth_info.json`. In a live
camera service, provide the same fields from the operator recipe instead of writing a file.

## Specialist model contracts

The offline plane pipeline accepts depth-refinement specialists under
`complex_scene.models`. Each specialist uses an existing refiner backend such as
TorchScript, TransCG DFNet, DREDS SwinDRNet, or ClearGrasp.

The industrial fixed-rail and CAD/pose runtimes accept task-specific contact-model
checkpoints with the same output contract as the current DTLD contact network. Multiple
scene names may share one checkpoint; it is loaded once.

Current server status:

- the existing object-pretrained metric depth-completion TorchScript model is
  retained for offline comparison, but both `glare` and `depth_failure` are
  disabled in the production configuration;
- the first scene-stratified pilot improved glare conditional MAE and coverage,
  but depth-failure, low-light, and general-transparent results failed their gates;
- `transparent_multilayer` V9.1 passes the synthetic candidate gate for selected
  liquid-interface depth at confidence 0.50, but remains disabled until real-camera
  domain, false-accept, plane-level height, and warm-latency qualification pass;
- newly generated product profiles contain empty specialist slots. The main product
  checkpoint remains active, and complex scenes are explicitly rejected until a qualified
  product specialist checkpoint is added.

This separation is intentional: transparent multi-layer ordering, RGB-D metric restoration,
and contact-curve prediction are different supervision targets.

## Latency and rejection

The deployment requirement is warm end-to-end P95 latency at or below 500 ms. The service
must remain resident; model loading and camera startup are reported separately from steady
measurement latency. Every result records total inference time, model time, budget, selected
variant, trigger reason, and whether the budget was met.

If a required specialist is unavailable, the result is rejected with
`complex_model_required_but_unavailable`. If the warm inference exceeds 500 ms, it is
rejected with `latency_budget_exceeded`. Neither condition silently falls back to a trusted
standard result.

The legacy plane-path regression currently measures about 242 ms mean and 274 ms P95 over
14 valid historical frames on the RTX 5090, with approximately 32 ms worst observed
depth-refinement time. This is a development-path measurement, not a camera-to-panel
certification. The product path must be benchmarked on the final camera stream.
The resident fixed-rail product smoke profile measured 41.5 ms mean and 41.8 ms
P95 over ten warm repetitions, including reference-motion checks; model prediction
was 8.3 ms mean/P95. It stayed well inside 500 ms, but this single historical frame
is only a runtime smoke test, not final accuracy or full live-camera certification.

## Promotion checklist

A specialist may be enabled only after all of the following pass on scene-disjoint data:

1. lower MAE and P95 liquid-depth/contact error than the standard model in its target scene;
2. no unacceptable regression on ordinary scenes when automatic routing is used;
3. false-accept ratio, accepted coverage, confidence calibration, and rejection reasons pass;
4. temporal jitter and camera-move behavior pass;
5. warm end-to-end P95 remains at or below 500 ms;
6. the checkpoint, dataset license, training configuration, and benchmark report are archived.

## Immediate algorithm work

1. Build one common scene-stratified evaluation manifest for DTLD/TCLD, TRADE real,
   LayeredDepth, TransCG, DREDS/STD, and ClearGrasp. Keep metric liquid-depth claims
   separate from auxiliary transparent-depth and layer-ordering metrics.
2. Train the transparent/multi-layer specialist: use full SeeGroup as an offline teacher,
   distill layer candidates into the lightweight ray-layer head, and select the liquid
   interface with metric RGB-D, container geometry, gravity, and temporal priors.
3. Train the glare/depth-failure specialist from the current object-pretrained checkpoint
   with balanced saturated-highlight, transparent, container-edge, and sensor-hole sampling.
4. Train a low-light specialist only after collecting or constructing a real dark validation
   split; start with exposure augmentation and bounded preprocessing rather than a large
   enhancement network.
5. Run selective-risk and 500 ms benchmarks. Promote only the variants that pass; leave the
   others disabled.
6. Collect a small project-specific industrial set with traceable liquid depths. Public
   datasets can validate components, but they cannot certify the requested approximately
   1% end-to-end metric accuracy for the actual camera, vessel, liquid, and lighting.

