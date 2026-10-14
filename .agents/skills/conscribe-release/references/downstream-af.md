# Downstream `af` propagation

Conscribe is consumed by the sibling repository at `/Users/qly/QLY/code/af`.

Only update that repository when the user includes it in scope. Before changing its dependency, confirm that no agent or verification run is active there; changing the environment mid-run invalidates its measurements.

After a successful Conscribe publication:

1. Update the Conscribe lower bound in `af/pyproject.toml` to the published version.
2. Run `uv sync` in the `af` repository.
3. Regenerate and validate affected generated config/stub outputs using that repository's own instructions. Never hand-edit its generated files.
4. Report the downstream diff and verification separately from the Conscribe release.
