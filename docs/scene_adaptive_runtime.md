# Scene-adaptive runtime and 500 ms deployment policy

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
- `transparent_multilayer` is disabled until SeeGroup/LayeredDepth distillation and
  metric validation pass;
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

