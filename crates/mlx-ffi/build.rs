// crates/mlx-ffi/build.rs
//
// Builds vendored mlx-c and MLX as static libraries.
//
// src/bindings.rs is checked in. Normal builds use that file without
// regenerating it.
//
// Environment switches:
//   MLX_FFI_REGENERATE=1
//       Regenerate src/bindings.rs using bindgen.
//
//   MLX_FFI_BINDGEN_ONLY=1
//       Requires MLX_FFI_REGENERATE=1. Skip CMake and native link
//       directives. Intended only for bindings regeneration/checks.
//
//   MLX_FFI_ASAN=1
//       Enable the existing CMake USE_ASAN option.
//
// Hermeticity: the CMake build never touches the network.
//   - mlx, json, fmt and (on macOS) metal-cpp are redirected to the
//     vendored trees in vendor/ via FETCHCONTENT_SOURCE_DIR_*.
//   - GGUF loading is disabled (MLX_BUILD_GGUF=OFF): it was the one
//     remaining FetchContent (gguflib) and is out of V1 scope.
//   - FETCHCONTENT_FULLY_DISCONNECTED=ON makes CMake fail at configure
//     time, naming the dependency, if anything unvendored is ever
//     requested again (for example after an MLX pin bump). That turns a
//     silent download into a loud error.
//
// Canonical container regeneration is driven by
// scripts/bindgen_check.sh.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate must live at <repo>/crates/mlx-ffi");

    let vendor = repo_root.join("vendor");
    let mlx_c = vendor.join("mlx-c");

    println!("cargo:rerun-if-env-changed=MLX_FFI_REGENERATE");
    println!("cargo:rerun-if-env-changed=MLX_FFI_BINDGEN_ONLY");
    println!("cargo:rerun-if-env-changed=MLX_FFI_ASAN");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", vendor.display());

    let regenerate = env::var_os("MLX_FFI_REGENERATE").is_some();
    let bindgen_only = env::var_os("MLX_FFI_BINDGEN_ONLY").is_some();

    if bindgen_only && !regenerate {
        panic!("MLX_FFI_BINDGEN_ONLY requires MLX_FFI_REGENERATE=1");
    }

    if !mlx_c.join("CMakeLists.txt").is_file() {
        panic!(
            "vendored mlx-c not found at {} — run scripts/vendor.sh first",
            mlx_c.display()
        );
    }

    let target_os =
        env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS must be set by Cargo");

    if !bindgen_only {
        build_and_link(&mlx_c, &vendor, &target_os);
    }

    let bindings_path = manifest_dir.join("src/bindings.rs");

    if regenerate {
        let bindings = bindgen::Builder::default()
            .header(
                manifest_dir
                    .join("wrapper.h")
                    .to_str()
                    .expect("wrapper.h path must be valid UTF-8"),
            )
            .clang_arg(format!("-I{}", mlx_c.display()))
            // Public mlx-c declarations and internal helpers such as
            // _mlx_array_is_available.
            .allowlist_function("_?mlx_.*")
            .allowlist_type("_?mlx_.*")
            .allowlist_var("MLX_.*")
            // FILE is used behind pointers in the mlx-c graph APIs.
            // Supply an opaque pointer target instead of its libc layout.
            .blocklist_type("FILE")
            // Darwin stdio implementation types.
            .blocklist_type("__sFILE")
            .blocklist_type("__sFILEX")
            .blocklist_type("__sbuf")
            .blocklist_type("fpos_t")
            .blocklist_type("__darwin_off_t")
            .blocklist_type("__int64_t")
            // glibc stdio implementation types.
            .blocklist_type("_IO_.*")
            .blocklist_type("__off_t")
            .blocklist_type("__off64_t")
            .raw_line(
                "#[repr(C)]\n\
                 #[derive(Debug)]\n\
                 pub struct FILE {\n\
                     _private: [u8; 0],\n\
                 }",
            )
            .formatter(bindgen::Formatter::Rustfmt)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("bindgen failed on wrapper.h; check vendored headers and Clang arguments");

        bindings
            .write_to_file(&bindings_path)
            .expect("failed to write src/bindings.rs");

        println!("cargo:warning=regenerated {}", bindings_path.display());
    } else if !bindings_path.is_file() {
        panic!(
            "src/bindings.rs is missing. Restore the checked-in file or \
             regenerate it with scripts/bindgen_check.sh --update, or \
             MLX_FFI_REGENERATE=1 cargo build -p mlx-ffi"
        );
    }
}

fn build_and_link(mlx_c: &Path, vendor: &Path, target_os: &str) {
    let on_macos = target_os == "macos";
    let on_linux = target_os == "linux";

    let mut cfg = cmake::Config::new(mlx_c);

    cfg.define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF");

    // Redirect the configured FetchContent dependencies to vendored trees.
    cfg.define("FETCHCONTENT_SOURCE_DIR_MLX", vendor.join("mlx"));
    cfg.define("FETCHCONTENT_SOURCE_DIR_JSON", vendor.join("json"));
    cfg.define("FETCHCONTENT_SOURCE_DIR_FMT", vendor.join("fmt"));

    if on_macos {
        cfg.define(
            "FETCHCONTENT_SOURCE_DIR_METAL_CPP",
            vendor.join("metal-cpp"),
        );
    }

    // Hermeticity guard. With this set, CMake refuses to download anything:
    // a FetchContent dependency that is not redirected above fails the
    // configure step with an error naming it, instead of silently using
    // the network.
    cfg.define("FETCHCONTENT_FULLY_DISCONNECTED", "ON");

    // GGUF loading is out of V1 scope, and gguflib (antirez/gguf-tools) was
    // the one MLX FetchContent we had not vendored. safetensors support is
    // a separate option (MLX_BUILD_SAFETENSORS) and stays at its default.
    cfg.define("MLX_BUILD_GGUF", "OFF");

    cfg.define("MLX_BUILD_CPU", "ON")
        .define("MLX_BUILD_CUDA", "OFF")
        .define("MLX_USE_CCACHE", "OFF");

    if on_macos {
        cfg.define("MLX_BUILD_METAL", "ON")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "14.0");
    } else if on_linux {
        cfg.define("MLX_BUILD_METAL", "OFF");
    }

    if env::var_os("MLX_FFI_ASAN").is_some() {
        cfg.define("USE_ASAN", "ON");
    }

    let dst = cfg.build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib64").display()
    );

    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    if on_macos {
        for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }

        println!("cargo:rustc-link-lib=c++");
    } else if on_linux {
        for library in ["dl", "pthread", "blas", "lapack", "stdc++"] {
            println!("cargo:rustc-link-lib=dylib={library}");
        }
    }
}
