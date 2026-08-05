//! Regenerate the type stubs for the `plugmem` package.
//!
//! Run it from the crate directory; CI runs it and fails on a `git diff`, the
//! same gate the Node binding's `index.d.ts` sits behind. A committed generated
//! file that nothing checks is a file that goes stale, and a stale `.pyi` is
//! worse than none: an IDE will confidently complete a method that no longer
//! exists.
//!
//! A binary rather than a build script because the information only exists once
//! the library is linked — the `gen_stub_*` macros register it through
//! `inventory`, which needs a real program to walk.

use std::path::Path;

/// Where `pyo3_stub_gen` puts the stub: a *package* directory, because that is
/// the shape it uses for a module that might have submodules.
const GENERATED_DIR: &str = "python/plugmem/_plugmem";

/// The committed stub, beside the compiled module as PEP 561 expects.
const STUB: &str = "python/plugmem/_plugmem.pyi";

/// Where it has to end up: a sibling of the compiled `_plugmem…so`.
///
/// `_plugmem` is an extension module, not a package. Leaving a directory of
/// that name next to the shared object invites the import machinery to resolve
/// the directory as a namespace package and shadow the real module — the stubs
/// would then be the only thing anyone could import. PEP 561 places a stub for
/// an extension module beside it, so that is where this puts it.
fn main() -> pyo3_stub_gen::Result<()> {
    _plugmem::stub_info()?.generate()?;

    let generated = Path::new(GENERATED_DIR).join("__init__.pyi");
    if generated.is_file() {
        std::fs::rename(&generated, STUB)?;
        // Only the directory the generator just made, and only if it is empty:
        // `remove_dir` refuses a non-empty one, so this can never become a
        // recursive delete of something that was already there.
        std::fs::remove_dir(GENERATED_DIR)?;
    }

    let stub = std::fs::read_to_string(STUB)?;
    let stub = unqualify_our_own_base(&stub);
    let stub = declare_the_error_code(&stub);
    std::fs::write(STUB, stub)?;
    println!("wrote {STUB}");
    Ok(())
}

/// Declare `code` on the exception base, which the generator cannot see.
///
/// `err.code` is API — it is how a caller branches without matching on prose,
/// and it is the string the Node binding puts on a thrown `Error`. But it is
/// attached at registration time as a class attribute, not through a getter,
/// so nothing in the macros knows about it and the stub omits it. A consumer
/// then gets `"LockedError" has no attribute "code"` from a type checker while
/// the attribute is right there at runtime.
///
/// Declared once on the base rather than twelve times on the subclasses: every
/// concrete error inherits it. The base's own runtime value is `None`, and
/// typing it `str` is still true of every instance that can exist, because
/// `PlugmemError` is only ever a catch target — every raise in this binding
/// uses a concrete subclass.
fn declare_the_error_code(stub: &str) -> String {
    const BASE: &str = "class PlugmemError(builtins.Exception):\n";
    match stub.find(BASE) {
        Some(at) => {
            let (head, tail) = stub.split_at(at + BASE.len());
            format!("{head}    code: typing.ClassVar[builtins.str]\n{tail}")
        }
        // The generator stopped emitting the base under this name: leave the
        // stub alone rather than write something that only looks right.
        None => stub.to_string(),
    }
}

/// Undo one wrong qualification the generator emits.
///
/// `pyo3_stub_gen`'s `create_exception!` describes an exception's base class
/// with `TypeInfo::builtin(name)`, which is right for `Exception` and wrong for
/// ours: it writes `class LockedError(builtins.PlugmemError)`, and there is no
/// such name in `builtins`. Left alone, `mypy --strict` reports 24 errors in
/// the stub and — worse, because it is silent — treats the whole exception
/// hierarchy as `Any`, so `except plugmem.LockedError` gets no checking at all
/// and an editor cannot show the tree.
///
/// The replacement is deliberately narrow: only our own base is unqualified,
/// so `builtins.Exception` (which is genuinely a builtin, and is
/// `PlugmemError`'s own base) is left exactly as it is.
fn unqualify_our_own_base(stub: &str) -> String {
    stub.replace("builtins.PlugmemError", "PlugmemError")
}
