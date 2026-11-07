//! Small, model-free installation check using the production safe MLX wrappers.
//!
//! Inputs are constructed once and retained while both explicit streams use
//! them. Cross-device operations pass the same native arrays directly, without
//! reconstructing arrays from host readback. Reading the final contiguous
//! outputs verifies the arithmetic and CPU access to Metal-produced storage.

use mlx_ffi::{Array, Device, DeviceType, Stream, ops};
use serde_json::{Value, json};

const INPUT: [f32; 4] = [1.0, -2.0, 3.0, -4.0];
const FACTOR: [f32; 4] = [0.5, 1.5, -2.0, 0.25];
const PRODUCT: [f32; 4] = [0.5, -3.0, -6.0, -1.0];
const CROSS_DEVICE: [f32; 4] = [1.5, -5.0, -3.0, -5.0];
const REQUIRE_METAL: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

fn checked<T>(operation: &str, result: Result<T, impl std::fmt::Display>) -> Result<T, String> {
    result.map_err(|error| format!("{operation}: {error}"))
}

fn stream(kind: DeviceType) -> Result<Stream, String> {
    let device = checked("create device", Device::new(kind, 0))?;
    if !checked("query device availability", device.is_available())? {
        return Err(format!("required {kind:?} device is unavailable"));
    }
    let stream = checked("create stream", Stream::new(&device))?;
    let actual = checked("query stream device", stream.device())?;
    if checked("query stream device kind", actual.kind())? != kind {
        return Err(format!(
            "native stream did not select required {kind:?} device"
        ));
    }
    Ok(stream)
}

fn verify(array: &Array, stream: &Stream, expected: &[f32; 4]) -> Result<[f32; 4], String> {
    if array.shape() != [2, 2] {
        return Err("native operation returned an unexpected shape".to_owned());
    }
    let values: [f32; 4] = checked("read native output", ops::to_vec_f32(array, stream))?
        .try_into()
        .map_err(|_| "native operation returned an unexpected element count".to_owned())?;
    if &values != expected {
        return Err("native operation returned incorrect values".to_owned());
    }
    Ok(values)
}

fn check(report: &mut Value) -> Result<(), String> {
    report["cpu"]["status"] = json!("failed");
    let cpu = stream(DeviceType::Cpu)?;
    let input = checked(
        "create shared input",
        Array::from_f32_slice(&INPUT, &[2, 2]),
    )?;
    let factor = checked(
        "create shared factor",
        Array::from_f32_slice(&FACTOR, &[2, 2]),
    )?;
    checked("evaluate shared input", input.eval())?;
    checked("evaluate shared factor", factor.eval())?;
    let input_address = checked("read shared input", input.data_f32())?.as_ptr();
    let factor_address = checked("read shared factor", factor.data_f32())?.as_ptr();

    let cpu_product = checked("CPU multiply", ops::multiply(&input, &factor, &cpu))?;
    let cpu_values = verify(&cpu_product, &cpu, &PRODUCT)?;
    report["cpu"] = json!({"status": "passed", "result": cpu_values});

    if !REQUIRE_METAL {
        return Ok(());
    }

    report["metal"]["status"] = json!("failed");
    let metal = stream(DeviceType::Gpu)?;
    let metal_product = checked("Metal multiply", ops::multiply(&input, &factor, &metal))?;
    let metal_values = verify(&metal_product, &metal, &PRODUCT)?;
    // This output is a dense elementwise result, so the supported data accessor
    // reads its contiguous native storage after evaluation/synchronization.
    let metal_address = checked("read Metal storage", metal_product.data_f32())?.as_ptr();
    report["metal"] = json!({"status": "passed", "required": true, "result": metal_values});

    report["shared_arrays"]["status"] = json!("failed");
    let cpu_to_metal = checked(
        "CPU-produced array on Metal",
        ops::residual_add(&cpu_product, &input, &metal),
    )?;
    let cpu_to_metal_values = verify(&cpu_to_metal, &metal, &CROSS_DEVICE)?;
    let metal_to_cpu = checked(
        "Metal-produced array on CPU",
        ops::residual_add(&metal_product, &input, &cpu),
    )?;
    let metal_to_cpu_values = verify(&metal_to_cpu, &cpu, &CROSS_DEVICE)?;

    let retained_input = checked("reread shared input", input.data_f32())?;
    let retained_factor = checked("reread shared factor", factor.data_f32())?;
    let retained_metal = checked("reread Metal storage", metal_product.data_f32())?;
    if retained_input.as_ptr() != input_address
        || retained_factor.as_ptr() != factor_address
        || retained_metal.as_ptr() != metal_address
        || retained_input != INPUT
        || retained_factor != FACTOR
        || retained_metal != PRODUCT
    {
        return Err("native shared-array storage or contents changed unexpectedly".to_owned());
    }
    report["shared_arrays"] = json!({
        "status": "passed",
        "input_storage_unchanged": true,
        "metal_output_cpu_readable": true,
        "cpu_to_metal_result": cpu_to_metal_values,
        "metal_to_cpu_result": metal_to_cpu_values,
    });
    Ok(())
}

/// Return one bounded report, including partial progress on a native failure.
pub(super) fn run() -> (bool, Value) {
    let unsupported = json!({
        "status": "unsupported",
        "reason": "requires_apple_silicon_macos",
    });
    let mut report = json!({
        "schema_version": 1,
        "version": env!("CARGO_PKG_VERSION"),
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "ok": false,
        "cpu": {"status": "not_run"},
        "metal": if REQUIRE_METAL {
            json!({"status": "not_run", "required": true})
        } else {
            json!({"status": "unsupported", "required": false, "reason": "requires_apple_silicon_macos"})
        },
        "shared_arrays": if REQUIRE_METAL { json!({"status": "not_run"}) } else { unsupported },
    });
    match check(&mut report) {
        Ok(()) => {
            report["ok"] = json!(true);
            (true, report)
        }
        Err(error) => {
            // Native status descriptions are short today; cap this diagnostic
            // even if their error implementation grows in a future pin.
            report["error"] = json!(error.chars().take(512).collect::<String>());
            (false, report)
        }
    }
}
