/* Copyright 2026 The alloy Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); ...
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bindgen input for crates/mlx-ffi. The mlx-c umbrella header includes
 * every public mlx-c header (all 27), so bindgen sees the full C API
 * surface — the same surface scripts/coverage_inventory.sh measures for
 * named test `coverage_inventory_complete`.
 *
 * The include path (vendor/mlx-c) is passed as a clang arg by build.rs
 * during regeneration (MLX_FFI_REGENERATE=1), never baked in here.
 *
 * Platform note: metal.h, cuda.h, and distributed.h must parse on the
 * Linux CPU CI lane even though the product is macOS-only. If any of
 * them fails to parse there, this file becomes a platform-conditional
 * include list — a CI-only concern. Recorded as an empirical checkpoint
 * in DR-0001.
 */

#include "mlx/c/mlx.h"