# Repository working instructions

## Scope and compatibility

- This is a Python video anomaly-localization research prototype plus a localhost demo.
- Read `README.md`, `docs/RUNNING.md`, and `docs/PROJECT_STATUS.md` before changing a runtime or evaluation entry point.
- Keep existing module imports, report schemas, API paths, FPS/time-range conventions, feature order, model signatures, and provenance validation compatible.
- Do not replace default models merely because a single experimental score improves. Preserve fixed grouping, OOF boundaries, negative-video and empty-candidate safeguards.
- Do not rename or reformat unrelated research files. Do not introduce production dependencies without authorization.

## Data and publication

- `.gitignore` intentionally admits only source directories and curated documentation. New root/doc paths require a deliberate publication decision.
- Never commit user videos/annotations, `.env`, credentials, third-party checkpoints, original portable trained bundles, caches, logs, or old reports with private provenance.
- Explicit owner-approved exception (2026-10-03): only the reviewed inference-only `models/public/locator.json` and `models/public/motion.json`, their registry and readout license may be published. Do not broaden this exception, copy raw model metadata, or admit new models without a separate review/authorization.
- Do not edit or rehash model metadata to bypass checkpoint, feature-profile, or provenance validation.
- Project-owned code is licensed under the root MIT `LICENSE`, as explicitly approved by the owner. Do not change that license without authorization or apply it to excluded third-party resources.
- Third-party copies require verified upstream license/provenance before redistribution.
- A publish request does not authorize public visibility, deployment, automatic workflows, external uploads for scanning, or history rewrites.

## Runtime and checks

- Frontend: `python code/demo_app.py`, localhost port 5002.
- GPU worker: `python -B code/optimized_model_worker.py --config LOCAL_TOML --serve` in the configured native/WSL model environment, localhost port 5004 by default. Read `docs/RESOURCE_SETUP.md`, the path policy and strict profile checks first; never download resources at app startup.
- Minimal demo and learned CPU packages are in `requirements-demo.txt`; `requirements-models.txt` plus the explicitly documented Torch environment cover product GPU paths, not all historical training experiments.
- Focused check: `python -m unittest discover -s code -p "test_demo.py" -v`.
- Portable check: `python -m unittest discover -s code -p "test_optimized_locator.py" -v`; report optional sklearn skips honestly.
- Broader tests may require local bundles, real feature caches, CUDA, or research dependencies. Never claim these checks passed if they did not run.
- Do not stop or restart an active demo while doing unrelated repository work.
