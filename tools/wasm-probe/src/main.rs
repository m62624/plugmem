//! Native runner: prints the scenario hash in the same decimal-`i64`
//! form wasmtime and wasmer print for an `--invoke run` result, so a CI
//! job can compare the three outputs with a string equality.
fn main() {
    println!("{}", plugmem_wasm_probe::scenario_hash() as i64);
}
