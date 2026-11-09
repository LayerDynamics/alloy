fn main() {
    println!("cargo:rerun-if-changed=native/fileport.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("native/fileport.c")
            .warnings(true)
            .compile("alloy_fileport");
        println!("cargo:rustc-link-lib=bsm");
    }
}
