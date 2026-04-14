/// Build script for the Python extension module.
///
/// On macOS, PyO3 extension modules are loaded by the Python interpreter at
/// runtime, so all `_Py*` symbols are provided by the interpreter itself.
/// The default linker behaviour on macOS rejects undefined symbols, which
/// causes the `cdylib` link to fail.  Passing `-undefined dynamic_lookup`
/// tells the linker to allow those symbols to be resolved at load time.
fn main() {
    // Only needed on macOS targets (both Apple Silicon and Intel).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
