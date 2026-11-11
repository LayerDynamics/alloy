fn main() {
    println!("cargo:rerun-if-changed=native/fileport.c");
    println!("cargo:rerun-if-changed=native/owned_process.c");
    println!("cargo:rerun-if-changed=native/owned_process.h");
    println!("cargo:rerun-if-changed=native/linux_supervisor.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        cc::Build::new()
            .file("native/linux_supervisor.c")
            .flag("-std=c11")
            .warnings(true)
            .compile("alloy_linux_supervisor");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("native/fileport.c")
            .file("native/owned_process.c")
            .flag("-std=c11")
            .warnings(true)
            .compile("alloy_fileport");
        println!("cargo:rustc-link-lib=bsm");
    }
}
