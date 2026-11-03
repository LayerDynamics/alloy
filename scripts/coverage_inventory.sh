#!/usr/bin/env bash
#
# scripts/coverage_inventory.sh — mlx-c coverage inventory and CI gate.
#
# Usage:
#   scripts/coverage_inventory.sh [--check]   gate: complete AND matches the
#                                             checked-in inventory (default)
#   scripts/coverage_inventory.sh --update    regenerate coverage/inventory.txt
#   scripts/coverage_inventory.sh --print     print the inventory to stdout
#   scripts/coverage_inventory.sh --test      run named tests 8 and 9
#
# Named tests:
#   coverage_inventory_complete  every public symbol in the vendored mlx-c
#                                headers has a Rust declaration, and the
#                                regenerated inventory equals the checked-in
#                                copy (the diff is empty)
#   coverage_drift_rejected      removing a symbol from the inventory, or a
#                                declaration from bindings.rs, makes the
#                                gate fail for the right reason. Only runs
#                                from a green baseline.
#
# Inventory rows: kind<TAB>symbol<TAB>header<TAB>status
#   kind    fn (C function) | type (typedef name)
#   status  declared    present in bindings.rs
#           missing     in the headers, absent from bindings.rs  -> FAILS
#           stale       fn in bindings.rs, absent from headers   -> FAILS
#           conditional declared behind HAS_FLOAT16 / HAS_BFLOAT16; whether
#                       bindgen sees it depends on the platform, so it is
#                       never required and its status never varies by host.
#                       This keeps the inventory byte-identical across
#                       macOS and the Linux CI image.
#
# Rust-side recognition: "pub fn", "pub type|struct|enum|union", and
# "pub use ... as <name>;" (bindgen re-exports typedef'd C enums as
# "pub use self::mlx_dtype_ as mlx_dtype;").
#
# The inventory header records vendor/mlx-c.sha256, so any pin bump changes
# the file and forces a deliberate --update (and review of the diff).
#
# Limits, stated plainly: headers are parsed with regexes, not a C parser.
# That relies on mlx-c's clang-format style (a function's name and its "("
# on one line). Under-extraction would hide a missing symbol on BOTH sides,
# so --check also requires a set of anchor symbols to be present.
#
# Overrides (used by --test to run the gate against mutated copies):
#   INVENTORY_HEADERS_DIR, INVENTORY_BINDINGS, INVENTORY_FILE
#
# Exit codes: 0 ok, 1 gate failure, 2 usage or environment error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO_ROOT/scripts/coverage_inventory.sh"

HEADERS_DIR="${INVENTORY_HEADERS_DIR:-$REPO_ROOT/vendor/mlx-c/mlx/c}"
BINDINGS="${INVENTORY_BINDINGS:-$REPO_ROOT/crates/mlx-ffi/src/bindings.rs}"
INVENTORY="${INVENTORY_FILE:-$REPO_ROOT/crates/mlx-ffi/coverage/inventory.txt}"
DIGEST_FILE="$REPO_ROOT/vendor/mlx-c.sha256"

# Symbols that must always be present as "fn ... declared". If header
# parsing silently broke, these would vanish.
ANCHORS="mlx_array_new mlx_closure_apply mlx_compile mlx_quantized_matmul"

die() { echo "coverage_inventory.sh: error: $*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need perl; need awk; need diff; need mktemp

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -d "$HEADERS_DIR" ] || die "headers not found: $HEADERS_DIR (run scripts/vendor.sh)"
[ -f "$BINDINGS" ]    || die "bindings not found: $BINDINGS"

# --- generation -------------------------------------------------------------

generate() {
  local digest="unknown"
  if [ -f "$DIGEST_FILE" ]; then
    digest="$(tr -d '[:space:]' < "$DIGEST_FILE")"
  fi
  perl - "$HEADERS_DIR" "$BINDINGS" "$digest" <<'PERL'
use strict;
use warnings;

my ($hdir, $bindings, $digest) = @ARGV;

opendir(my $dh, $hdir) or die "cannot open $hdir: $!\n";
my @headers = sort grep { /\.h$/ && -f "$hdir/$_" } readdir $dh;
closedir $dh;

# "kind\0name" => [first header, conditional?]
my %hdr;
for my $h (@headers) {
    open my $fh, '<', "$hdir/$h" or die "cannot read $hdir/$h: $!\n";
    my $src = do { local $/; <$fh> };
    close $fh;

    $src =~ s{/\*.*?\*/}{ }gs;     # block comments
    $src =~ s{//[^\n]*}{}g;        # line comments

    my @stack;                      # 1 = inside a HAS_*FLOAT16 conditional
    for my $line (split /\n/, $src) {
        if ($line =~ /^\s*#\s*(?:if|ifdef|ifndef)\b(.*)/) {
            push @stack, ($1 =~ /HAS_B?FLOAT16/ ? 1 : 0);
            next;
        }
        next if $line =~ /^\s*#\s*(?:elif|else)\b/;
        if ($line =~ /^\s*#\s*endif\b/) { pop @stack; next; }
        next if $line =~ /^\s*#/;

        my $cond = (grep { $_ } @stack) ? 1 : 0;
        my @found;
        while ($line =~ /(?<!\w)(_?mlx_\w+)\s*\(/g) {
            push @found, ['fn', $1];
        }
        if ($line =~ /^\s*\}\s*(mlx_\w+)\s*;/) {
            push @found, ['type', $1];
        } elsif ($line =~ /^\s*typedef\b.*?\b(mlx_\w+)\s*;/) {
            push @found, ['type', $1];
        }
        for my $f (@found) {
            my $key = "$f->[0]\0$f->[1]";
            if (exists $hdr{$key}) {
                $hdr{$key}[1] = 0 unless $cond;   # any unconditional decl wins
            } else {
                $hdr{$key} = [$h, $cond];
            }
        }
    }
}

my (%rs_fn, %rs_type);
open my $bf, '<', $bindings or die "cannot read $bindings: $!\n";
while (my $l = <$bf>) {
    if ($l =~ /^\s*pub\s+fn\s+(_?mlx_\w+)/) {
        $rs_fn{$1} = 1;
    } elsif ($l =~ /^\s*pub\s+(?:type|struct|enum|union)\s+(mlx_\w+)/) {
        $rs_type{$1} = 1;
    } elsif ($l =~ /^\s*pub\s+use\s+.*\bas\s+(mlx_\w+)\s*;/) {
        $rs_type{$1} = 1;      # e.g. pub use self::mlx_dtype_ as mlx_dtype;
    }
}
close $bf;

print "# mlx-ffi coverage inventory\n";
print "# generated by scripts/coverage_inventory.sh --update; do not hand-edit\n";
print "# mlx-c content digest: $digest\n";
print "# columns: kind<TAB>symbol<TAB>header<TAB>status\n";

for my $key (sort keys %hdr) {
    my ($kind, $name) = split /\0/, $key;
    my ($h, $cond) = @{ $hdr{$key} };
    my $have = $kind eq 'fn' ? $rs_fn{$name} : $rs_type{$name};
    my $status = $cond ? 'conditional' : ($have ? 'declared' : 'missing');
    print join("\t", $kind, $name, $h, $status), "\n";
}
for my $name (sort keys %rs_fn) {
    next if exists $hdr{"fn\0$name"};
    print join("\t", 'fn', $name, '-', 'stale'), "\n";
}
PERL
}

# --- gate -------------------------------------------------------------------

# cmd_check: returns 0 when complete and drift-free, 1 otherwise.
cmd_check() {
  generate > "$TMP/generated.txt"

  local sym
  for sym in $ANCHORS; do
    if ! awk -F'\t' -v s="$sym" '$1=="fn" && $2==s && $4=="declared" {f=1} END{exit !f}' \
         "$TMP/generated.txt"; then
      echo "FAIL: anchor symbol '$sym' not extracted as declared;" >&2
      echo "      header parsing may have broken, or bindings.rs is stale." >&2
      return 1
    fi
  done

  local bad
  bad="$(awk -F'\t' '!/^#/ && ($4=="missing" || $4=="stale")' "$TMP/generated.txt")"
  if [ -n "$bad" ]; then
    echo "FAIL: bindings.rs does not match the vendored mlx-c headers:" >&2
    printf '%s\n' "$bad" | sed -n '1,40p' >&2
    echo "  (kind, symbol, header, status; first 40 shown)" >&2
    return 1
  fi

  if [ ! -f "$INVENTORY" ]; then
    echo "FAIL: no checked-in inventory at $INVENTORY; run --update." >&2
    return 1
  fi
  if ! diff -u "$INVENTORY" "$TMP/generated.txt" > "$TMP/drift.diff"; then
    echo "FAIL: inventory drift (checked-in copy differs from regenerated):" >&2
    sed -n '1,60p' "$TMP/drift.diff" >&2
    echo "  If the pin or bindings changed on purpose, run --update and review." >&2
    return 1
  fi
  echo "ok: inventory complete and matches $INVENTORY"
}

cmd_update() {
  generate > "$TMP/generated.txt"
  local bad
  bad="$(awk -F'\t' '!/^#/ && ($4=="missing" || $4=="stale")' "$TMP/generated.txt")"
  if [ -n "$bad" ]; then
    echo "refusing to write an incomplete inventory:" >&2
    printf '%s\n' "$bad" | sed -n '1,40p' >&2
    return 1
  fi
  mkdir -p "$(dirname "$INVENTORY")"
  cp "$TMP/generated.txt" "$INVENTORY"
  echo "wrote $INVENTORY"
  awk -F'\t' '!/^#/ {c[$1 " " $4]++} END {for (k in c) print "  " k ": " c[k]}' \
    "$INVENTORY" | LC_ALL=C sort
}

# --- named tests 8 and 9 ----------------------------------------------------
#
# NOTE: cmd_test runs under "cmd_test || exit 1" in the dispatcher, which
# disables errexit inside it. Every command whose failure matters is
# therefore checked explicitly below.

cmd_test() {
  local fail=0

  # Test 8: coverage_inventory_complete
  if bash "$SELF" --check > "$TMP/t8.out" 2> "$TMP/t8.err"; then
    echo "ok   coverage_inventory_complete"
  else
    echo "FAIL coverage_inventory_complete"
    cat "$TMP/t8.err"
    echo "SKIP coverage_drift_rejected: mutation tests need a green baseline"
    return 1
  fi

  # Test 9 only means something from a passing baseline (guaranteed above).
  # A non-anchor symbol is mutated so the completeness path, not the anchor
  # guard, is what rejects it.
  local victim="mlx_array_free"

  # 9a: a symbol removed from the inventory is rejected as drift.
  awk -F'\t' -v v="$victim" '!($1=="fn" && $2==v)' "$INVENTORY" \
    > "$TMP/inv-mutated.txt" || { echo "FAIL coverage_drift_rejected: awk failed"; return 1; }
  if cmp -s "$INVENTORY" "$TMP/inv-mutated.txt"; then
    echo "FAIL coverage_drift_rejected: inventory has no entry for $victim"
    fail=1
  elif INVENTORY_FILE="$TMP/inv-mutated.txt" bash "$SELF" --check \
         >/dev/null 2> "$TMP/t9a.err"; then
    echo "FAIL coverage_drift_rejected: gate accepted an inventory missing $victim"
    fail=1
  elif ! grep -q 'inventory drift' "$TMP/t9a.err" || ! grep -q "$victim" "$TMP/t9a.err"; then
    echo "FAIL coverage_drift_rejected: rejected, but not as drift on $victim:"
    cat "$TMP/t9a.err"
    fail=1
  else
    echo "ok   coverage_drift_rejected (inventory entry removed)"
  fi

  # 9b: a declaration removed from bindings.rs is rejected as missing.
  grep -v "pub fn ${victim}(" "$BINDINGS" > "$TMP/bindings-mutated.rs" || true
  if cmp -s "$BINDINGS" "$TMP/bindings-mutated.rs"; then
    echo "FAIL coverage_drift_rejected: bindings.rs has no declaration of $victim"
    fail=1
  elif INVENTORY_BINDINGS="$TMP/bindings-mutated.rs" bash "$SELF" --check \
         >/dev/null 2> "$TMP/t9b.err"; then
    echo "FAIL coverage_drift_rejected: gate accepted bindings missing $victim"
    fail=1
  elif ! grep -q "$victim" "$TMP/t9b.err" || ! grep -q 'missing' "$TMP/t9b.err"; then
    echo "FAIL coverage_drift_rejected: rejected, but not as missing $victim:"
    cat "$TMP/t9b.err"
    fail=1
  else
    echo "ok   coverage_drift_rejected (declaration removed)"
  fi

  return "$fail"
}

# --- dispatch ---------------------------------------------------------------

case "${1:---check}" in
  --check)  cmd_check || exit 1 ;;
  --update) cmd_update || exit 1 ;;
  --print)  generate ;;
  --test)   cmd_test || exit 1 ;;
  -h|--help) sed -n '2,48p' "$0" ;;
  *) die "unknown option: $1 (try --help)" ;;
esac