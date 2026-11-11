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
        let mut native = cc::Build::new();
        native
            .file("native/fileport.c")
            .file("native/owned_process.c")
            .flag("-std=c11")
            .warnings(true);
        if std::env::var("PROFILE").as_deref() == Ok("debug") {
            native.define("ALLOY_OWNED_PROCESS_TEST", None);
        }
        native.compile("alloy_fileport");
        println!("cargo:rustc-link-lib=bsm");
        println!("cargo:rerun-if-changed=native/helper_main.c");
        if std::env::var("PROFILE").as_deref() == Ok("debug") {
            let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
            for (name, pipe) in [
                ("alloy-plugin-pipe-helper", true),
                ("alloy-plugin-exec-helper", false),
            ] {
                let compiler = cc::Build::new().get_compiler();
                let mut command = compiler.to_command();
                command
                    .args([
                        "-std=c11",
                        "native/helper_main.c",
                        "native/fileport.c",
                        "native/owned_process.c",
                        "-lbsm",
                        "-o",
                    ])
                    .arg(out.join(name));
                if pipe {
                    command.arg("-DALLOY_PIPE_HELPER");
                }
                assert!(
                    command
                        .status()
                        .expect("compile development helper")
                        .success(),
                    "development helper build failed"
                );
            }
            println!(
                "cargo:rustc-env=ALLOY_DEV_HELPER={}",
                out.join("alloy-plugin-pipe-helper").display()
            );
        }
    }
}
