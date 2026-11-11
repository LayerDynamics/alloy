//! Isolated macOS supervisor: own child reaping and clean its process group.
#[cfg(target_os = "macos")]
fn main() {
    #[link(name = "alloy_fileport", kind = "static")]
    unsafe extern "C" {
        fn alloy_exec_helper(argc: i32, argv: *mut *mut std::ffi::c_char) -> i32;
    }
    use std::os::unix::ffi::OsStrExt;
    let args: Vec<_> = std::env::args_os()
        .map(|s| std::ffi::CString::new(s.as_bytes()).expect("OS argument contains NUL"))
        .collect();
    let mut pointers: Vec<_> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
    pointers.push(std::ptr::null_mut());
    // SAFETY: all C strings and the null-terminated pointer array remain live.
    std::process::exit(unsafe { alloy_exec_helper(args.len() as i32, pointers.as_mut_ptr()) });
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the exec helper is supported only on macOS");
    std::process::exit(2);
}
