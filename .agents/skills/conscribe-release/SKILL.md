---
name: conscribe-release
description: Prepare and publish a Conscribe version when the user asks for a release, version bump, or PyPI publication. Covers documentation, version synchronization, review gating, release push, uv build, and Twine upload; do not use for ordinary development, branch-only pushes, or local builds.
---

# Release Conscribe

Carry a reviewed Conscribe change through its complete release. The workflow is repository-specific; do not replace it with a generic tagging or CI release flow.

## Non-negotiable invariants

- Use uv for the project environment. Run tests and Python entry points through `uv run`.
- Publish with Twine, never `uv publish`.
- Never add a `Co-Authored-By` trailer.
- Do not create a git tag. This repository releases without tags.
- Never read, print, copy, or commit a PyPI token. Twine obtains credentials from the user's existing configuration.
- Never upload `dist/*`: the directory retains older, already-published artifacts. Name only the two artifacts for the new version.
- Build and check the artifacts before the approval gate. Never push a release commit whose artifacts have not passed the Twine check and the artifact-content check.
- Treat `git push` and PyPI upload as distinct external actions. A request for only one never implies the other.
- Show the prepared diff, the checked artifacts, and a plain-language release summary before external actions. The user's explicit approval of a summary that names both push and upload is the single gate for both; after it, complete push and upload without asking again.

## Prepare the release

1. Confirm the intended semantic version and inspect the branch, upstream, and working tree. Preserve unrelated user changes. Query PyPI for that exact version and stop if it already exists; PyPI releases and files cannot be overwritten.
2. Make the requested implementation and tests.
3. Synchronize the version in `pyproject.toml` and `conscribe/__init__.py`, then run `uv lock` and `uv lock --check` so the local project entry in `uv.lock` carries the same version. Also update the current-version text in both `AGENTS.md` and `CLAUDE.md`.
4. Update `CHANGELOG.md`, `README.md`, `conscribe/llms.txt`, and every affected file under `conscribe/docs/`. Documentation must describe observable behavior, compatibility impact, and any known limitation; do not make empty version-only edits to unaffected prose.
5. Run `uv run pytest`. Resolve failures before continuing.
6. Review the complete diff and repository status. Ensure no secrets, generated caches, unrelated files, or old `dist/` artifacts are staged.
7. Commit the release with the repository's conventional prefix (`fix:`, `feat:`, or `chore:`) and suffix `— v<version>`. Do not add a co-author trailer.
8. Require a clean tracked working tree, then build fresh artifacts from the release commit with `uv build`. Record that commit and the SHA-256 of the two new artifacts.
9. Validate only the new artifacts. Use an ephemeral, major-version-bounded Twine installation so the workflow also works in a fresh clone:

   ```bash
   uv run --with 'twine>=6,<7' twine check \
     dist/conscribe-<version>-py3-none-any.whl \
     dist/conscribe-<version>.tar.gz
   ```

10. Check that the artifacts ship exactly what the commit tracks. The `conscribe/` files in both the wheel and the sdist must equal `git ls-files conscribe`, so docs and `llms.txt` cannot go missing and untracked files cannot slip in. Besides those, the sdist holds exactly the allowlisted `pyproject.toml`, `README.md`, `CHANGELOG.md`, and `LICENSE`, plus the `PKG-INFO` and `.gitignore` that hatchling adds itself:

    ```bash
    uv run python - <version> <<'PY'
    import subprocess, sys, tarfile, zipfile
    version = sys.argv[1]
    ls = subprocess.run(["git", "ls-files", "-z", "conscribe"], check=True, capture_output=True, text=True)
    tracked = set(ls.stdout.split("\0")) - {""}
    with zipfile.ZipFile(f"dist/conscribe-{version}-py3-none-any.whl") as whl:
        wheel = {n for n in whl.namelist() if not n.startswith(f"conscribe-{version}.dist-info/")}
    with tarfile.open(f"dist/conscribe-{version}.tar.gz") as tar:
        sdist = {m.name.split("/", 1)[1] for m in tar.getmembers() if m.isfile()}
    top_level = {"pyproject.toml", "README.md", "CHANGELOG.md", "LICENSE", "PKG-INFO", ".gitignore"}
    failed = False
    for label, got, want in (("wheel", wheel, tracked), ("sdist", sdist, tracked | top_level)):
        if got != want:
            failed = True
            print(f"{label}: missing {sorted(want - got)}, unexpected {sorted(got - want)}")
    print("artifacts do not match the commit" if failed else f"artifacts match the commit ({len(tracked)} package files)")
    sys.exit(1 if failed else 0)
    PY
    ```

    A mismatch blocks the release. Report it; never loosen the check, and never delete the user's untracked files to make it pass.

11. Give the user a plain-language description of the behavior change, validation results (tests, Twine check, artifact check), version/doc updates, commit, the two checked artifacts with their SHA-256, and the exact actions that approval will trigger. Wait for approval.

## Publish after approval

Use the latest explicit user-requested order:

1. Require a clean tracked working tree and confirm that `HEAD` is the reviewed release commit. If you cannot confirm that the artifacts in `dist/` were built from `HEAD` and still have the recorded SHA-256, repeat preparation steps 8–10 before pushing; otherwise do not rebuild.
2. Push the reviewed commit to its configured upstream with `git push`, then verify that the upstream branch resolves to the same commit as local `HEAD`.
3. Recheck that PyPI still has no files for the target version, then upload the already-checked artifacts by those same exact paths non-interactively:

   ```bash
   uv run --with 'twine>=6,<7' twine upload --non-interactive \
     dist/conscribe-<version>-py3-none-any.whl \
     dist/conscribe-<version>.tar.gz
   ```

4. Query PyPI and verify that the release contains exactly the expected wheel and sdist. Compare their published hashes with the local artifacts and report the pushed branch, commit, version, filenames, hashes, and PyPI result.

If push, build, validation, or upload fails, stop the remaining dependent steps and report the exact failure. If an upload partially succeeds, inspect PyPI's file list and upload only a confirmed-missing artifact; never retry both blindly and never hide the situation with `--skip-existing`.

## Optional downstream propagation

When the user also asks to update the sibling `af` repository, read [references/downstream-af.md](references/downstream-af.md). Treat it as a separate scoped follow-up, not implicit permission from a Conscribe release request.
