/**
 * plugmem-wasm — the WebAssembly build of the plugmem memory engine.
 *
 * Stage-5 stub surface: version + companion-skill accessors. The engine
 * class (open/remember/recall/revise/forget/link/maintain over storage and
 * embedder callbacks, specs/06) lands in a later release and will extend
 * this module without breaking the exports below.
 */

/** Engine/package version (tracks the workspace release), e.g. `"0.1.0"`. */
export function version(): string;

/** Short, version-free description pointing at the skill. */
export function about(): string;

/**
 * The companion skill for wasm consumers: the canonical `SKILL.md` with the
 * CLI/MCP "Run it" appendix stripped (a wasm host ships skill and engine
 * from the same release, so the transport/version ceremony never applies).
 */
export function skill(): string;

/** The canonical, unstripped `SKILL.md` (what CLI/MCP consumers read). */
export function skillFull(): string;

/** The `<!-- skill-version: X.Y.Z -->` marker value from the canonical skill. */
export function skillVersion(): string;
