// Native unified-memory discovery bridge for Apple-Silicon macOS.
fn main() {
    println!("cargo:rerun-if-changed=native/unified_memory.mm");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo target architecture");

    if target_os != "macos" || target_arch != "aarch64" {
        return;
    }

    cc::Build::new()
        .cpp(true)
        .file("native/unified_memory.mm")
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-mmacosx-version-min=15.0")
        .warnings(true)
        .compile("alloy_unified_platform");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
