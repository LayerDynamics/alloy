//! Minimal macOS pipe broker. It never starts or supervises a plugin.
#[cfg(target_os = "macos")]
fn main() {
    alloy_process_owner::link_native_supervisor();
    #[link(name = "alloy_fileport", kind = "static")]
    #[link(name = "bsm")]
    unsafe extern "C" {
        fn alloy_fileport_helper() -> i32;
    }
    if std::env::args_os().count() != 2
        || std::env::args_os().nth(1).as_deref()
            != Some(std::ffi::OsStr::new("--alloy-pipe-helper-v1"))
    {
        std::process::exit(2);
    }
    // SAFETY: the helper is single-threaded and the function has no arguments.
    std::process::exit(unsafe { alloy_fileport_helper() });
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the pipe helper is supported only on macOS");
    std::process::exit(2);
}
