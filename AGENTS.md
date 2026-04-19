# Agent instructions for `general_tracking/`

This repository is a downstream project built on top of `mjlab`.

- Depend on `mjlab` as a package, not via a local path dependency.
- Keep custom tasks, robots, data processing, and learning code in this repository.
- Use the local `controller/mjlab` checkout only as a source reference unless upstream work is explicitly required.

