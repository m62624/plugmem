# `plugmem-darwin-arm64`

The prebuilt **aarch64-apple-darwin** (macOS arm64 (Apple Silicon)) native addon for
[`plugmem`](https://www.npmjs.com/package/plugmem) — an embedded long-term
memory engine for local-first applications and agents (remember / recall / revise / forget over one
local database).

## Don't install this directly

This is a platform-specific binary. Install the main package instead:

```sh
npm install plugmem
```

`plugmem` lists every `plugmem-*` binary in its `optionalDependencies`, and npm
installs only the one matching your OS and CPU — this package is the one picked
on **macOS arm64 (Apple Silicon)**. You never depend on it by name.

## Links

- Main package — <https://www.npmjs.com/package/plugmem>
- Source, docs & issues — <https://github.com/m62624/plugmem>

MIT licensed.
