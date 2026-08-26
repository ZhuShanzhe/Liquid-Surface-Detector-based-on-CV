# Software copyright registration and productization plan

## Goal and boundary

The planned registrable product is provisionally named **RGB-D工业液深智能测量系统 V1.0**. The
registration object is the project's independently developed software expression and its documentation:
RGB-D frame ingestion, liquid-surface perception adapters, metric geometry, uncertainty/quality gating,
temporal fusion, command-line workflow, evaluation, and deployment utilities.

Registration is not a patent, an algorithm novelty review, a measurement-system certification, or proof that
the system reaches a particular accuracy. The accuracy profile remains a product verification requirement.

## Official material baseline

Under the Chinese Computer Software Copyright Registration Measures, an application includes the application
form, software identification material, and relevant proof documents. Program and document identification
materials normally contain the first and last 30 consecutive pages; if either has fewer than 60 pages, submit
the whole material. Unless an exceptional deposit is used, program pages normally contain at least 50 lines
and document pages at least 30 lines. Application material is in Chinese on A4 paper.

The Software Protection Regulations describe documentation as material explaining content, composition,
design, functional specifications, development, test results, and usage. They protect independently developed
program expression and documentation, not the underlying ideas, processing methods, operations, or
mathematical concepts.

Primary references:

- [Computer Software Copyright Registration Measures](https://www.ncac.gov.cn/xxfb/flfg/bmgz/)
- [Computer Software Protection Regulations](https://xzfg.moj.gov.cn/front/law/detail?LawID=914)
- [China Copyright Protection Center](https://www.ccopyright.com.cn/)

This document is an engineering checklist, not legal advice. The applicant should verify the live portal
requirements before filing.

## Registration source boundary

Include only project-owned source and configuration needed to explain the product:

- src/liquid_depth: core perception, geometry, confidence, temporal, and evaluation code;
- scripts: original training, evaluation, audit, export, and data-contract tools;
- ros2_ws/src/liquid_depth_camera: the project's camera capture adapter;
- configs: runtime contracts, accuracy profile, and product metadata.

Exclude from original source deposit:

- third_party/OrbbecSDK_ROS2 and all other upstream repositories;
- downloaded datasets, papers, external model implementations, and third-party pretrained weights;
- Conda environments, CUDA/ROS installations, caches, generated artifacts, logs, and captured customer data;
- model weight binaries even when trained by the project; record their provenance separately.

Released external algorithms may be reproduced for research, but their source must not be presented as
project-owned code. Only independently written adapters, losses, fusion logic, calibration, and product
workflow belong to the candidate deposit. Preserve every required upstream license and notice in deployment
packages.

## Product quality specification

The machine-readable acceptance policy is configs/accuracy_profile_industrial_v1.yaml. It separates the
measured liquid depth from camera standoff and uses a hybrid tolerance:

    allowed error = max(absolute millimeter floor, relative percentage * reference liquid depth)

The product reports MAE, signed bias, RMSE, P95, temporal jitter, accepted coverage, confidence, and rejection
reason. A result rejected by the quality gate is not counted as a successful measurement.

The research target is high accuracy, including approximately 1% MAE from 1 to 10 m where the qualified sensor
configuration supports it. A wider deployment gate is retained for controlled rollout, but no mode may be
marketed at an accuracy that has not been verified with traceable references across material, lighting,
temperature, distance, and camera-mode strata.

## Work packages and release gates

### P0: ownership and naming freeze

- Confirm Chinese full name, short name, V1.0 version, applicant, copyright holder, completion date, and
  publication status in configs/software_product_v1.yaml.
- Review employment, university, commissioned-development, and collaboration agreements.
- Decide the root repository distribution license. The ROS package currently declares MIT while the root has
  no license; resolve this inconsistency before V1.0.
- Keep credentials, cookies, private data, and personal identifiers outside Git and filing material.

### P1: registrable product core

- Provide one documented command from synchronized camera or recorded replay to liquid-depth JSON output.
- Freeze stable input/output schemas, semantic versioning, diagnostics, and rejection reasons.
- Keep optional research backends behind interfaces so V1.0 runs without copying third-party repositories.
- Add deterministic installation, health check, demo/replay data contract, and uninstall/cleanup instructions.

### P2: algorithm completion

- Use TransCG/DREDS/ClearGrasp only as baselines and pretraining sources.
- Promote a project-owned multi-task model for liquid mask, restored metric depth or geometric offset, normal,
  and uncertainty.
- Add calibrated container geometry, meniscus handling, gravity/bottom constraints, and robust temporal fusion.
- Stratify training and validation for transparent, translucent, glare, saturated highlight, and container edge.

### P3: industrial verification

- Build traceable reference captures across the 0.2-10 m primary range; retain DTLD as a 20-200 mm stress test.
- Qualify every camera model, lens, resolution, exposure/emitter mode, mounting pose, and container family.
- Run at least three independent sessions and 30 repeats per static condition.
- Publish target and deployment-gate reports with scenario coverage and false-accept analysis.
- Do not claim 1% at 5-10 m until the selected sensor configuration actually passes.

### P4: filing artifact freeze

- Complete a Chinese user manual or design specification with screenshots, architecture, operation, output,
  error handling, test results, and version identification.
- Generate the candidate source inventory and SHA-256 manifest with scripts/audit_copyright_release.py.
- Select the consecutive source and document pages only after final formatting; preserve file order and line
  continuity.
- Tag the exact reviewed commit as v1.0.0-copyright and archive source, manual, inventory, test report, and build
  instructions together.
- Submit applicant identity and ownership proof through the official system; these cannot be generated by code.

## Current readiness

The project already has an end-to-end architecture, camera adapter, classical baseline, learned-model
interfaces, robust geometry, temporal filtering, scenario evaluation, server bootstrap, and more than 5,000
lines of project source. It is not filing-ready yet because ownership metadata and the distribution-license
decision are unresolved, the user manual is still a draft, final industrial-range validation data does not
exist, and the V1.0 algorithm has not passed the piecewise acceptance profile.
