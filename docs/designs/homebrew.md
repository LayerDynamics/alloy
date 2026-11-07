# Homebrew installation

## Scope and availability

Homebrew is Alloy's primary installation path on Apple Silicon. The first
formula packages the existing CLI, daemon and unprivileged monitor together
from one immutable Git commit. It is a development preview using the project's
`0.1.0` version, not a claim that a tagged release or bottle has been published.
The source change and the corresponding
[tap formula](https://github.com/LayerDynamics/homebrew-alloy/blob/main/Formula/alloy.rb)
must both merge before the public installation command works.

This brings a usable source formula forward from the distribution roadmap.
It does not publish the reserved npm `alloyml` package or any Cargo package,
create a daemon service, or complete release automation. Phase 1 acceptance
is already [recorded separately](../acceptance/phase-1-exit-review.md).
Publishing the telemetry privacy policy and operating a collection endpoint
remain the existing prerequisites for a tagged emitting pre-release. This
installation does not invent an endpoint or claim those prerequisites are met.

## User workflow

Requires Homebrew, Apple Silicon, macOS 14 or newer, and Xcode 15 or newer with
its command-line and Metal compiler tools. The initial source build compiles
vendored MLX and can take several minutes. A missing compiler is an installation error with
build diagnostics, not a reason to defer compilation until the first chat.

```bash
brew install LayerDynamics/alloy/alloy
alloy --version
alloyd --check-native
alloy start
```

The fully qualified name selects this project's formula. Homebrew maps the
`LayerDynamics/alloy` tap to `LayerDynamics/homebrew-alloy`; another tap's
formula with the same short name is a different package. An existing `alloy`
executable can still cause a link conflict. Resolve its owning installation
explicitly; installation must not force an overwrite of unrelated commands.

Installation downloads no model, writes no user configuration, and launches
no daemon. `alloy start` keeps the existing explicit local-model/download
choice and foreground lifetime. Wait for `READY — Alloy API 1`, then use
`alloy chat` or `alloy monitor` in another terminal. `alloy setup --download`
is the separate explicit starter-model provisioning command. See
[startup.md](startup.md) for model verification and readiness limits.

## Installed payload and toolchains

Paths below are relative to this formula's versioned installation prefix.
The `bin` entry points select private runtime files rather than the user's
Node, Cargo, working directory, or a neighboring source checkout.

| Location | Payload and purpose |
| --- | --- |
| `bin/alloy` | CLI entry point using the managed Node runtime and paired daemon default. |
| `bin/alloyd` | Entry point for the matching native daemon. |
| `libexec/cli/` | CLI launcher, compiled JavaScript, package metadata and `models/starter-model.json`. |
| `libexec/cli/dist/native/alloy-metrics` | Unprivileged Apple Silicon sampler at the relative path expected by the compiled CLI. |
| `libexec/native/alloyd` | Actual native daemon executable. |
| `libexec/native/mlx.metallib` | MLX Metal kernels adjacent to the executable, available after the build tree is removed. |
| `libexec/node/` | Private Node runtime, managed with this formula. |

| Tool | Pin and role |
| --- | --- |
| Node | `26.13.0`, official `darwin-arm64` distribution with publisher SHA-256; used to build the CLI and retained as its runtime. |
| Rust | `1.99.1`, official `aarch64-apple-darwin` distribution with publisher SHA-256; build-only, matching `rust-toolchain.toml`. |
| CMake | Homebrew build dependency for vendored MLX. |
| Apple compiler/SDK | Xcode 15 or newer, including command-line tools and the Metal compiler; native macOS deployment target 14.0. |

The formula owns exact toolchain resources because a moving Homebrew `node`
or `rust` dependency does not preserve Alloy's patch-version pins. It must not
install a global rustup toolchain or change the user's Node selection. npm
dependencies and Rust crates follow their checked-in lockfiles. Source
fetching includes the pinned vendored native dependencies; no runtime code is
downloaded on the first invocation.

Static MLX linkage does not embed `mlx.metallib`. Vendored MLX first searches
beside its binary, then other resource locations and a build-time fallback.
Packaging only `alloyd` can therefore appear to work while the build directory
exists and fail later. The adjacent kernel library is required production
payload, and validation must exercise it after the build tree is unavailable.

The installed CLI retains explicit daemon-setting and `ALLOYD` overrides.
Its default is the paired daemon; a compatible daemon already listening on
the chosen socket is still reused according to the existing startup contract.
Monitoring invokes neither `sudo` nor Apple's privileged `powermetrics` tool.
Unsupported hardware counters remain explicitly unavailable.

## Upgrades and user data

Settings, models, the SQLite session database, Unix socket and persisted KV
cache belong outside the Cellar. Their default root is `~/.alloy/`;
`ALLOY_CONFIG_DIR` selects another root. Existing explicit paths in saved
settings remain authoritative. Use the same environment setting in the
startup, chat and monitor terminals.

```bash
brew update
brew upgrade LayerDynamics/alloy/alloy
```

Upgrading replaces installed files without restarting an existing daemon.
Stop the old foreground `start` with Ctrl+C and run `alloy start` again to
use the new daemon. `Already running` means the compatible listener was
reused; it does not prove that process runs the newly installed executable.
Upgrade and uninstall preserve external user data.

When an untagged source pin changes, the formula's version or revision must
advance so `brew upgrade` selects the new build. Source identity, CLI/daemon
versions and the formula revision belong in the installation evidence.

## Validation and publication gate

Before treating the formula as available, require the actual Apple Silicon
source install and its installed-payload checks:

1. Build the immutable source with the pinned resources and install all
   runtime files. Record the source commit, formula revision, macOS/SDK and
   toolchain versions, and complete build output.
2. Run installed CLI/daemon version and help commands, `alloyd --check-native`
   for both CPU and Metal, and `alloy monitor --once --json` with isolated
   configuration. The check must fail if Metal cannot execute; CPU success
   alone is insufficient.
3. Remove access to build outputs, then verify installed native execution and
   generation with the existing frozen starter artifact. Confirm the CLI
   launches the paired installed daemon, streams a terminal response, and
   retains settings and session state outside the Cellar. Model files are
   pre-provisioned for this check, not downloaded by installation.
4. Merge the source change, pin the tested source in the tap, and merge the
   matching formula. Record actual outcomes before changing the availability
   notice. These checks establish this installation path; they do not claim
   bottles, npm distribution, a hosted telemetry service, or Phase 2 tools.

The source repository's physical Mac job runs
[`scripts/homebrew/ci.mjs`](../../scripts/homebrew/ci.mjs). It creates a unique
temporary tap and a keg-only copy of the generated formula, changing only the
formula name/class, adding `keg_only`, and selecting an owned local source URL.
[`source.mjs`](../../scripts/homebrew/source.mjs) creates an independent bare
snapshot of the exact checked-out commit and tree, with a stable `candidate`
ref and verified complete objects. This supports detached/shallow synthetic
PR merge commits that GitHub may not serve as ordinary download revisions;
the immutable commit pin is unchanged. The production fetch, install and
test bodies stay identical. Actual `brew install --build-from-source` and
`brew test` run against that owned formula. Its exclusive Homebrew build
directory is moved out of the original path before native checks and the
installed real-model smoke. Commands use the owned prefix and isolated
configuration; the job never links over the user's `alloy` command. Cleanup
removes only that unique keg/tap, with automatic update, cleanup and dependency
autoremove disabled. The generated formula, source/installation identities,
build/test logs and real-model result are retained as acceptance evidence.

The canonical tap retains its HTTPS GitHub source URL. Its hosted ARM64 macOS
job validates that public source download, builds the canonical `alloy`
formula and checks normal command linking and CLI/daemon selection. It does
not run the native acceptance test or claim Metal execution on the hosted VM.
That packaging result complements the required physical CPU/Metal/shared-array
and model run; neither substitutes for the other.

Homebrew's [tap documentation](https://docs.brew.sh/Taps),
[Formula Cookbook](https://docs.brew.sh/Formula-Cookbook), and
[language-specific guidance](https://docs.brew.sh/Language-Specific-Formulae)
define the tap naming, private payload layout and dependency conventions.
