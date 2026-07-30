# `plugmem-linux-arm64-gnu`

The prebuilt **aarch64-unknown-linux-gnu** (Linux arm64 (glibc)) native addon for
[`plugmem`](https://www.npmjs.com/package/plugmem) — an embedded long-term
memory engine for LLM agents (remember / recall / revise / forget over one
local file).

## Don't install this directly

This is a platform-specific binary. Install the main package instead:

```sh
npm install plugmem
```

`plugmem` lists every `plugmem-*` binary in its `optionalDependencies`, and npm
installs only the one matching your OS and CPU — this package is the one picked
on **Linux arm64 (glibc)**. You never depend on it by name.

## Links

- Main package — <https://www.npmjs.com/package/plugmem>
- Source, docs & issues — <https://github.com/m62624/plugmem>

MIT licensed.
