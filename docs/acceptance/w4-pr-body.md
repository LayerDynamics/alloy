## W4 builtin execution, cancellation and scoped acceptance

Refs #31

Concrete builtin adapters are delivered. This change adds cross-session cleanup
checks, named W4 gates, and Linux/macOS evidence capture in the existing native CI.
See `docs/acceptance/w4-exit-review.md`; platform evidence and review are pending.

P2-09 here covers W4 builtins. W5/W6 generation/approval/hook pause integration and
external caller lock discipline remain separate. No Phase 2 sign-off is claimed.

Keep this PR draft until evidence and review are complete. Change Refs #31 to
Closes #31 only after recording the scoped W4 acceptance decision.
