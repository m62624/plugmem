# `plugmem-win32-arm64-msvc`

The prebuilt **aarch64-pc-windows-msvc** (Windows arm64 (MSVC)) native addon for
[`plugmem`](https://www.npmjs.com/package/plugmem). It is an embedded long-term
memory engine for local-first applications and agents (remember / recall / revise / forget over one
local database).

## Don't install this directly

This is a platform-specific binary. Install the main package instead:

```sh
npm install plugmem
```

`plugmem` lists every `plugmem-*` binary in its `optionalDependencies`, and npm
installs only the one matching your OS and CPU. This package is the one picked
on **Windows arm64 (MSVC)**. You never depend on it by name.

## Links

- Main package — <https://www.npmjs.com/package/plugmem>
- Source, docs & issues — <https://github.com/m62624/plugmem>

MIT licensed.
