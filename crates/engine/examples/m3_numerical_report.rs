//! Build the numerical sections of an M3 qualification report from real native captures.
//! Usage: m3_numerical_report PROFILE PINNED_REFERENCE RAW_ROWS NEW_OUTPUT_DIRECTORY
use engine::conformance::{NumericalMetrics, NumericalObservation, NumericalReference};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Pin {
    id: String,
    reference: ReferencePin,
    prompt_tokens: Vec<Vec<u32>>,
    decode_tokens: Vec<u32>,
}
#[derive(Deserialize)]
struct ReferencePin {
    sha256: String,
}
#[derive(Deserialize)]
struct ReferenceFile {
    profile: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    tokens: Vec<u32>,
    decode_tokens: Vec<u32>,
    logits: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRow {
    profile: String,
    reference_sha256: String,
    case: usize,
    step: usize,
    tokens: Vec<u32>,
    logits: Vec<f64>,
}

fn pinned(profile: &str) -> Result<(Pin, usize)> {
    let (manifest, vocabulary) = match profile {
        "gemma-4-e4b-it-4bit" => (
            include_str!("../../inference-infra/tests/fixtures/gemma4_e4b_profile.json"),
            262_144,
        ),
        "ornith-1.5-9b-4bit" => (
            include_str!("../../inference-infra/tests/fixtures/ornith_1_5_9b_profile.json"),
            248_320,
        ),
        "qwen3.5-0.8b-4bit" => (
            include_str!("../../inference-infra/tests/fixtures/qwen35_0_8b_profile.json"),
            248_320,
        ),
        _ => return Err(format!("no pinned numerical suite for {profile}").into()),
    };
    let pin: Pin = serde_json::from_str(manifest)?;
    if pin.id != profile {
        return Err("profile pin mismatch".into());
    }
    Ok((pin, vocabulary))
}

fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() > limit {
        return Err(format!("invalid input size: {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("input grew beyond size limit".into());
    }
    Ok(bytes)
}

fn input_digest(reference_sha: &str, tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"alloy-m3-numerical-input-v1\0");
    hash.update(reference_sha.as_bytes());
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(Default)]
struct Sections {
    references: BTreeMap<String, NumericalReference>,
    observations: Vec<NumericalObservation>,
    metrics: BTreeMap<String, NumericalMetrics>,
}

impl Sections {
    fn load(pin: &Pin, vocabulary: usize, frozen: ReferenceFile, dir: &Path) -> Result<Self> {
        if frozen.profile != pin.id || frozen.cases.len() != pin.prompt_tokens.len() {
            return Err("reference case/profile mismatch".into());
        }
        let mut filenames = fs::read_dir(dir)?
            .map(|e| e.map(|f| f.file_name()))
            .collect::<std::io::Result<BTreeSet<OsString>>>()?;
        let mut result = Self::default();
        for (case_index, case) in frozen.cases.into_iter().enumerate() {
            if case.tokens != pin.prompt_tokens[case_index]
                || case.decode_tokens != pin.decode_tokens
                || case.logits.len() != pin.decode_tokens.len() + 1
            {
                return Err("reference tokens or row count differ from profile".into());
            }
            let mut prefix = case.tokens;
            for (step, expected) in case.logits.into_iter().enumerate() {
                if step > 0 {
                    prefix.push(pin.decode_tokens[step - 1]);
                }
                let filename = format!("case-{case_index}-step-{step}.json");
                if !filenames.remove(&OsString::from(&filename)) {
                    return Err(format!("missing row {filename}").into());
                }
                let row: NativeRow = serde_json::from_slice(&read(&dir.join(filename), 16 << 20)?)?;
                if row.profile != pin.id
                    || row.reference_sha256 != pin.reference.sha256
                    || row.case != case_index
                    || row.step != step
                    || row.tokens != prefix
                {
                    return Err(
                        format!("native input mismatch at case {case_index}, step {step}").into(),
                    );
                }
                if expected.len() != vocabulary || row.logits.len() != vocabulary {
                    return Err(format!(
                        "wrong vocabulary width at case {case_index}, step {step}"
                    )
                    .into());
                }
                let id = format!("case-{case_index}-step-{step}");
                let input_sha256 = input_digest(&pin.reference.sha256, &prefix);
                let metrics = NumericalMetrics::compare(&expected, &row.logits)
                    .map_err(|e| format!("{id}: {e}"))?;
                result.metrics.insert(id.clone(), metrics);
                result.references.insert(
                    id.clone(),
                    NumericalReference {
                        input_sha256: input_sha256.clone(),
                        logits: expected,
                    },
                );
                result.observations.push(NumericalObservation {
                    fixture_id: id,
                    input_sha256,
                    logits: row.logits,
                });
            }
        }
        if !filenames.is_empty() {
            return Err(format!("unexpected capture files: {filenames:?}").into());
        }
        Ok(result)
    }

    fn write(&self, dir: &Path) -> Result<()> {
        fs::create_dir(dir)?;
        write(&dir.join("references.json"), &self.references)?;
        write(&dir.join("numerical.json"), &self.observations)?;
        write(&dir.join("numerical-metrics.json"), &self.metrics)?;
        if !self.metrics.values().all(NumericalMetrics::passes) {
            return Err("FR-8 failed; per-row measurements retained".into());
        }
        Ok(())
    }
}

fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut out = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer(&mut out, value)?;
    out.flush()?;
    out.get_ref().sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [profile, reference, raw, out] = args.as_slice() else {
        return Err(
            "usage: m3_numerical_report PROFILE PINNED_REFERENCE RAW_ROWS NEW_OUTPUT_DIRECTORY"
                .into(),
        );
    };
    let (pin, vocabulary) = pinned(profile.to_str().ok_or("invalid profile name")?)?;
    let bytes = read(Path::new(reference), 64 << 20)?;
    if format!("{:x}", Sha256::digest(&bytes)) != pin.reference.sha256 {
        return Err("reference SHA-256 mismatch".into());
    }
    let frozen = serde_json::from_slice(&bytes)?;
    let sections = Sections::load(&pin, vocabulary, frozen, Path::new(raw))?;
    sections.write(Path::new(out))?;
    println!(
        "{}: {} native numerical rows passed FR-8",
        pin.id,
        sections.metrics.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Dir(std::path::PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn dir() -> Dir {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let index = NEXT.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "alloy-numerical-{}-{stamp}-{index}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Dir(path)
    }
    fn pin() -> Pin {
        Pin {
            id: "fixture".into(),
            reference: ReferencePin {
                sha256: "a".repeat(64),
            },
            prompt_tokens: vec![vec![1, 2]],
            decode_tokens: vec![],
        }
    }
    fn reference() -> ReferenceFile {
        ReferenceFile {
            profile: "fixture".into(),
            cases: vec![Case {
                tokens: vec![1, 2],
                decode_tokens: vec![],
                logits: vec![vec![1.0, 0.0]],
            }],
        }
    }
    fn row() -> serde_json::Value {
        json!({"profile":"fixture","reference_sha256":"a".repeat(64),"case":0,"step":0,"tokens":[1,2],"logits":[1.0,0.0]})
    }
    #[test]
    fn m3_numerical_report_rejects_missing_swapped_and_nonfinite_rows() {
        let dir = dir();
        let file = dir.0.join("case-0-step-0.json");
        let check = || Sections::load(&pin(), 2, reference(), &dir.0);
        assert!(check().is_err());
        for (key, bad) in [
            ("tokens", json!([2, 1])),
            ("case", json!(1)),
            ("step", json!(1)),
            ("profile", json!("other")),
            ("reference_sha256", json!("b".repeat(64))),
            ("logits", json!([null, 0])),
            ("logits", json!([1.0])),
        ] {
            let mut value = row();
            value[key] = bad;
            fs::write(&file, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(check().is_err(), "{key}");
        }
        fs::write(&file, serde_json::to_vec(&row()).unwrap()).unwrap();
        assert!(check().is_ok());
        fs::write(dir.0.join("unexpected.json"), "{}").unwrap();
        assert!(check().is_err());
    }
    #[test]
    fn m3_numerical_report_preserves_native_float_bits() {
        // Observed E4B case 0, step 0, vocabulary index 2.
        let decimal = "-10.223902702331543";
        let native = f64::from(decimal.parse::<f32>().unwrap());
        let raw = dir();
        let mut value = row();
        value["logits"] = json!([native, 0.0]);
        write(&raw.0.join("case-0-step-0.json"), &value).unwrap();
        let mut frozen = reference();
        frozen.cases[0].logits[0][0] = serde_json::from_str(decimal).unwrap();
        let sections = Sections::load(&pin(), 2, frozen, &raw.0).unwrap();
        assert_eq!(
            sections.references["case-0-step-0"].logits[0].to_bits(),
            native.to_bits()
        );
        assert_eq!(
            sections.observations[0].logits[0].to_bits(),
            native.to_bits()
        );
    }

    #[test]
    fn m3_numerical_report_retains_failed_metrics_and_refuses_overwrite() {
        let raw = dir();
        let output = dir();
        let file = raw.0.join("case-0-step-0.json");
        let mut value = row();
        value["logits"] = json!([1.1, 0.001]);
        write(&file, &value).unwrap();
        let sections = Sections::load(&pin(), 2, reference(), &raw.0).unwrap();
        let dest = output.0.join("report");
        assert!(sections.write(&dest).is_err());
        let metrics: BTreeMap<String, NumericalMetrics> =
            serde_json::from_slice(&fs::read(dest.join("numerical-metrics.json")).unwrap())
                .unwrap();
        assert!(!metrics["case-0-step-0"].passes());
        let obs: Vec<NumericalObservation> =
            serde_json::from_slice(&fs::read(dest.join("numerical.json")).unwrap()).unwrap();
        assert_eq!(obs[0].logits, [1.1, 0.001]);
        fs::write(&file, serde_json::to_vec(&row()).unwrap()).unwrap();
        let good = Sections::load(&pin(), 2, reference(), &raw.0).unwrap();
        assert!(good.write(&dest).is_err());
        good.write(&output.0.join("second")).unwrap();
    }
}
