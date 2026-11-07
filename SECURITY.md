# Security Policy

## Supported versions

Alloy is pre-1.0 and moves quickly. Security fixes land on `main`; once tagged
releases exist, only the latest release is supported. Run a current checkout or
the latest Homebrew tap commit.

## Reporting a vulnerability

Report privately through GitHub: **Security → Advisories → Report a
vulnerability** on `LayerDynamics/alloy`. Please do not open public issues for
vulnerabilities.

Include the affected commit or version, your chip and macOS version,
reproduction steps or a proof of concept, and the impact you see. Expect an
acknowledgment within 7 days and a triage decision within 30 days.

## Scope and threat model

In scope and treated as serious:

- Tool-policy engine bypasses, approval-flow circumvention, or effect-class
  confusion (read/write/destructive).
- Crossing OS user boundaries. Note that `alloyd` listens on a Unix domain
  socket enforced by filesystem permissions; any process running as the *same*
  user can drive policy-gated requests — that is a documented residual risk
  (spec §3.7), not a vulnerability.
- Unexpected network egress. The telemetry client (FR-35) is the only
  always-on outbound connection; model downloads and user-configured MCP are
  user-initiated. Anything else phoning home is a bug.
- MCP annotation spoofing leading to unintended tool execution, and escapes
  from the plugin sandbox profiles.
- Model artifact or supply-chain integrity issues: artifacts are verified by
  manifest hash before load; vendored mlx-c/MLX is pinned with hashes in
  `PINS.md`. Upstream CVEs affecting the pinned commits are welcome.
- CI/workflow injection, including anything affecting the persistent
  self-hosted macOS runner.

Out of scope: issues requiring physical access, social engineering, or
vulnerabilities in third-party dependencies with no demonstrated impact on
Alloy.
