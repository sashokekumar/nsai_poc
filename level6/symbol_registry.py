# level6/symbol_registry.py
"""
SymbolRegistry -- persistent lifecycle store for evolved symbols.

Each symbol is born from a failure cluster (Task 7/8), passes through a
two-stage validation gate (Tasks 11/11B), and is either promoted into the
Level 5 rule_base or deprecated and eventually garbage-collected.

Schema for each symbol entry
-----------------------------
symbol_id               : str   "S_001", "S_002", ...
name                    : str   auto-derived by SymbolCluster (_symbol_name)
status                  : str   one of SymbolStatus values
created_cycle           : int   evolution cycle when symbol was registered
last_evaluated_cycle    : int   most recent cycle in which any evaluation ran

Grounding (from SymbolCluster)
  predicate_profile     : dict  {present, absent, uncertain, centroid}
  coverage              : int   number of failure-set members in source cluster
  cohesion              : float
  grounding_quality     : float
  dominant_confusion    : str   e.g. "investigate -> summarization"
  majority_gold_intent  : str
  majority_predicted_intent : str
  example_utterances    : list[str]
  low_confidence_grounding : bool  True when grounding_quality < 0.50

Candidate rule (set by RuleCandidateGen, Task 10)
  candidate_rule_name   : str | None
  rule_strength_init    : float | None

Validation (set by RuleValidator, Tasks 11/11B)
  accuracy_delta_noretrain    : float | None  Task 11
  false_positive_rate         : float | None  Task 11
  retrain_delta_over_noretrain: float | None  Task 11B

Weakening / deprecation bookkeeping
  consecutive_weak_cycles : int
  weakening_count         : int
  deprecated_cycle        : int | None

Usage
-----
    from level6.symbol_registry import SymbolRegistry

    registry = SymbolRegistry()                       # loads from disk if exists
    registry.register_from_cluster(cluster_dict)      # born from SymbolCluster output
    registry.promote("S_001")                         # Proposed -> Experimental
    registry.promote("S_001")                         # Experimental -> Active
    registry.tick_cycle()                             # advance cycle counter
    registry.garbage_collect()                        # remove old Deprecated entries
    registry.save()                                   # persist to disk

    # CLI: register clusters from clusters.json
    python -m level6.symbol_registry --register level6/data/clusters.json
    python -m level6.symbol_registry --status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from level6.lifecycle import (  # noqa: E402
    SymbolStatus,
    PROPOSED_TO_EXPERIMENTAL,
    GARBAGE_COLLECTION,
    can_promote_to_experimental,
    can_promote_to_active,
    should_weaken,
    should_deprecate,
    should_garbage_collect,
)

DEFAULT_REGISTRY_PATH = REPO_ROOT / "level6" / "data" / "symbol_registry.json"
_LOW_GROUNDING_THRESHOLD = PROPOSED_TO_EXPERIMENTAL["min_grounding_quality"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SymbolRegistry:
    """
    JSON-backed store for all evolved symbols, with a full lifecycle state machine.

    The registry is a dict keyed by symbol_id ("S_001", ...).  It also tracks
    the current evolution cycle counter.

    Thread safety: single-process, file-locked by atomic save (write + rename).
    """

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH):
        self.path   = path
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Coerce status strings back to SymbolStatus enum
            for sym in raw.get("symbols", {}).values():
                sym["status"] = SymbolStatus(sym["status"])
            return raw
        return {"current_cycle": 0, "symbols": {}}

    def save(self):
        """Persist registry to disk atomically (write tmp, rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise SymbolStatus enum values to plain strings
        payload = {
            "current_cycle": self._data["current_cycle"],
            "symbols": {
                sid: {**sym, "status": sym["status"].value}
                for sid, sym in self._data["symbols"].items()
            },
        }
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(self.path)

    # ------------------------------------------------------------------ #
    # ID generation
    # ------------------------------------------------------------------ #

    def _next_id(self) -> str:
        existing = [
            int(k[2:]) for k in self._data["symbols"]
            if k.startswith("S_") and k[2:].isdigit()
        ]
        n = (max(existing) + 1) if existing else 1
        return f"S_{n:03d}"

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_from_cluster(self, cluster: dict) -> str | None:
        """
        Register a new symbol from a SymbolCluster output dict.

        Noise clusters (is_noise_cluster=True) are rejected silently.
        Returns the new symbol_id, or None if registration was rejected.
        """
        if cluster.get("is_noise_cluster"):
            return None

        sid = self._next_id()
        gq  = cluster.get("grounding_quality", 0.0)

        self._data["symbols"][sid] = {
            "symbol_id":   sid,
            "name":        cluster["symbol_name"],
            "status":      SymbolStatus.PROPOSED,
            "created_cycle":        self._data["current_cycle"],
            "last_evaluated_cycle": None,

            # Grounding
            "predicate_profile": {
                "present":   cluster.get("present_predicates",   []),
                "absent":    cluster.get("absent_predicates",    []),
                "uncertain": cluster.get("uncertain_predicates", []),
                "centroid":  cluster.get("centroid", {}),
            },
            "coverage":              cluster["size"],
            "cohesion":              cluster["cohesion"],
            "grounding_quality":     gq,
            "dominant_confusion":    cluster.get("dominant_confusion", ""),
            "majority_gold_intent":       cluster.get("majority_gold_intent", ""),
            "majority_predicted_intent":  cluster.get("majority_predicted_intent", ""),
            "example_utterances":    cluster.get("example_utterances", []),
            "low_confidence_grounding": gq < _LOW_GROUNDING_THRESHOLD,

            # Candidate rule (filled by RuleCandidateGen, Task 10)
            "candidate_rule_name":  None,
            "rule_strength_init":   None,

            # Validation metrics (filled by RuleValidator, Tasks 11/11B)
            "accuracy_delta_noretrain":     None,
            "false_positive_rate":          None,
            "retrain_delta_over_noretrain": None,

            # Weakening / deprecation bookkeeping
            "consecutive_weak_cycles": 0,
            "weakening_count":         0,
            "deprecated_cycle":        None,
        }
        return sid

    def register_from_clusters_file(self, clusters_path: Path) -> list[str]:
        """
        Register all non-noise clusters from a clusters.json file.
        Skips clusters whose symbol_name already exists in the registry.
        Returns list of newly registered symbol_ids.
        """
        with open(clusters_path, encoding="utf-8") as fh:
            clusters = json.load(fh)

        existing_names = {s["name"] for s in self._data["symbols"].values()}
        new_ids: list[str] = []
        for cluster in clusters:
            if cluster.get("is_noise_cluster"):
                continue
            if cluster["symbol_name"] in existing_names:
                print(f"  [skip] '{cluster['symbol_name']}' already registered")
                continue
            sid = self.register_from_cluster(cluster)
            if sid:
                new_ids.append(sid)
                print(f"  [registered] {sid} = {cluster['symbol_name']}  "
                      f"(size={cluster['size']}, gq={cluster['grounding_quality']:.4f})")
        return new_ids

    # ------------------------------------------------------------------ #
    # Lifecycle transitions
    # ------------------------------------------------------------------ #

    def _get(self, symbol_id: str) -> dict:
        sym = self._data["symbols"].get(symbol_id)
        if sym is None:
            raise KeyError(f"Symbol '{symbol_id}' not found in registry.")
        return sym

    def promote(self, symbol_id: str) -> tuple[bool, str]:
        """
        Attempt to promote a symbol to the next lifecycle state.

        Proposed     -> Experimental  (gate: can_promote_to_experimental)
        Experimental -> Active        (gate: can_promote_to_active)

        Returns (success, message).
        """
        sym = self._get(symbol_id)
        current = sym["status"]

        if current == SymbolStatus.PROPOSED:
            ok, reasons = can_promote_to_experimental(sym)
            if not ok:
                return False, "Cannot promote to Experimental: " + "; ".join(reasons)
            sym["status"] = SymbolStatus.EXPERIMENTAL
            sym["last_evaluated_cycle"] = self._data["current_cycle"]
            return True, f"{symbol_id} promoted: Proposed -> Experimental"

        if current == SymbolStatus.EXPERIMENTAL:
            ok, reasons = can_promote_to_active(sym)
            if not ok:
                return False, "Cannot promote to Active: " + "; ".join(reasons)
            sym["status"] = SymbolStatus.ACTIVE
            sym["last_evaluated_cycle"] = self._data["current_cycle"]
            return True, f"{symbol_id} promoted: Experimental -> Active"

        if current in (SymbolStatus.ACTIVE, SymbolStatus.WEAKENING):
            return False, f"{symbol_id} is already {current} — no further promotion."

        if current == SymbolStatus.DEPRECATED:
            return False, f"{symbol_id} is Deprecated and cannot be promoted."

        return False, f"Unknown status '{current}' for {symbol_id}."

    def update_validation(self, symbol_id: str, **kwargs):
        """
        Update validation metrics on a symbol after running Tasks 11/11B.

        Accepted kwargs:
            accuracy_delta_noretrain    : float
            false_positive_rate         : float
            retrain_delta_over_noretrain: float
            candidate_rule_name         : str
            rule_strength_init          : float
        """
        sym = self._get(symbol_id)
        allowed = {
            "accuracy_delta_noretrain",
            "false_positive_rate",
            "retrain_delta_over_noretrain",
            "candidate_rule_name",
            "rule_strength_init",
        }
        for key, val in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Unknown validation field: '{key}'")
            sym[key] = val
        sym["last_evaluated_cycle"] = self._data["current_cycle"]

    def deprecate(self, symbol_id: str) -> tuple[bool, str]:
        """Manually deprecate a symbol (also used by evolution engine)."""
        sym = self._get(symbol_id)
        if sym["status"] == SymbolStatus.DEPRECATED:
            return False, f"{symbol_id} is already Deprecated."
        sym["status"] = SymbolStatus.DEPRECATED
        sym["deprecated_cycle"] = self._data["current_cycle"]
        return True, f"{symbol_id} deprecated at cycle {self._data['current_cycle']}."

    # ------------------------------------------------------------------ #
    # Cycle management
    # ------------------------------------------------------------------ #

    def tick_cycle(self) -> int:
        """
        Advance the evolution cycle counter by 1.
        Also runs automatic Weakening and Deprecation transitions for all
        Active/Weakening symbols based on their current accuracy_delta.
        Returns the new cycle number.
        """
        self._data["current_cycle"] += 1
        cycle = self._data["current_cycle"]

        for sym in self._data["symbols"].values():
            status = sym["status"]

            # Active -> Weakening check
            if status == SymbolStatus.ACTIVE:
                acc_delta = sym.get("accuracy_delta_noretrain") or 0.0
                from level6.lifecycle import WEAKENING_CRITERIA
                if acc_delta < WEAKENING_CRITERIA["max_accuracy_delta"]:
                    sym["consecutive_weak_cycles"] = sym.get("consecutive_weak_cycles", 0) + 1
                else:
                    sym["consecutive_weak_cycles"] = 0
                if should_weaken(sym):
                    sym["status"] = SymbolStatus.WEAKENING
                    sym["weakening_count"] = 1

            # Weakening -> Deprecated check
            elif status == SymbolStatus.WEAKENING:
                sym["weakening_count"] = sym.get("weakening_count", 0) + 1
                if should_deprecate(sym):
                    sym["status"] = SymbolStatus.DEPRECATED
                    sym["deprecated_cycle"] = cycle

        return cycle

    def garbage_collect(self) -> list[str]:
        """
        Remove Deprecated symbols that have exceeded the GC grace period.
        Returns list of removed symbol_ids.
        """
        cycle    = self._data["current_cycle"]
        grace    = GARBAGE_COLLECTION["deprecated_age_cycles"]
        to_remove = [
            sid for sid, sym in self._data["symbols"].items()
            if should_garbage_collect(sym, cycle)
            and (cycle - (sym.get("deprecated_cycle") or 0)) >= grace
        ]
        for sid in to_remove:
            del self._data["symbols"][sid]
        return to_remove

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    @property
    def current_cycle(self) -> int:
        return self._data["current_cycle"]

    def all_symbols(self) -> list[dict]:
        return list(self._data["symbols"].values())

    def by_status(self, status: SymbolStatus) -> list[dict]:
        return [s for s in self._data["symbols"].values() if s["status"] == status]

    def get(self, symbol_id: str) -> dict:
        return self._get(symbol_id)

    def summary(self) -> dict:
        counts = {s.value: 0 for s in SymbolStatus}
        for sym in self._data["symbols"].values():
            counts[sym["status"].value] += 1
        return {
            "current_cycle": self._data["current_cycle"],
            "total_symbols": len(self._data["symbols"]),
            "by_status": counts,
        }

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def print_status(self):
        summ = self.summary()
        print()
        print("=" * 60)
        print("  SymbolRegistry Status")
        print("=" * 60)
        print(f"  Evolution cycle  : {summ['current_cycle']}")
        print(f"  Total symbols    : {summ['total_symbols']}")
        print()
        print("  By lifecycle state:")
        for state, count in summ["by_status"].items():
            bar = "#" * count
            print(f"    {state:<14}  {count:3d}  {bar}")
        print()

        for sym in self.all_symbols():
            status    = sym["status"].value
            gq        = sym.get("grounding_quality", 0.0)
            lcg_flag  = "  [LOW_GROUNDING]" if sym.get("low_confidence_grounding") else ""
            ad        = sym.get("accuracy_delta_noretrain")
            fpr       = sym.get("false_positive_rate")
            rule_name = sym.get("candidate_rule_name") or "-"

            ad_str  = f"{ad:.4f}"  if ad  is not None else "n/a"
            fpr_str = f"{fpr:.4f}" if fpr is not None else "n/a"

            print(f"  {sym['symbol_id']}  [{status}]{lcg_flag}")
            print(f"      name          : {sym['name']}")
            print(f"      coverage      : {sym['coverage']}  "
                  f"cohesion={sym['cohesion']:.4f}  "
                  f"grounding_quality={gq:.4f}")
            print(f"      confusion     : {sym.get('dominant_confusion', '')}")
            print(f"      gold intent   : {sym.get('majority_gold_intent', '')}  "
                  f"predicted: {sym.get('majority_predicted_intent', '')}")
            print(f"      candidate rule: {rule_name}")
            print(f"      acc_delta     : {ad_str}  fpr: {fpr_str}")
            print(f"      created cycle : {sym['created_cycle']}  "
                  f"last eval: {sym.get('last_evaluated_cycle')}")
            print()
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SymbolRegistry CLI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--register", type=Path, metavar="CLUSTERS_JSON",
        help="Register non-noise clusters from clusters.json into the registry.",
    )
    g.add_argument(
        "--status", action="store_true",
        help="Print current registry state.",
    )
    g.add_argument(
        "--promote", metavar="SYMBOL_ID",
        help="Attempt to promote SYMBOL_ID to the next lifecycle state.",
    )
    g.add_argument(
        "--tick", action="store_true",
        help="Advance the evolution cycle counter and run auto-transitions.",
    )
    g.add_argument(
        "--gc", action="store_true",
        help="Run garbage collection on Deprecated symbols.",
    )
    p.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
        help="Path to symbol_registry.json (default: level6/data/symbol_registry.json)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    registry = SymbolRegistry(args.registry)

    if args.register:
        print(f"[SymbolRegistry] Registering clusters from: {args.register}")
        new_ids = registry.register_from_clusters_file(args.register)
        if new_ids:
            registry.save()
            print(f"[SymbolRegistry] Registered {len(new_ids)} new symbol(s): {new_ids}")
            print(f"[SymbolRegistry] Registry saved -> {args.registry}")
        else:
            print("[SymbolRegistry] No new symbols registered.")
        registry.print_status()

    elif args.status:
        registry.print_status()

    elif args.promote:
        ok, msg = registry.promote(args.promote)
        print(f"[SymbolRegistry] {msg}")
        if ok:
            registry.save()

    elif args.tick:
        new_cycle = registry.tick_cycle()
        removed   = registry.garbage_collect()
        registry.save()
        print(f"[SymbolRegistry] Cycle advanced to {new_cycle}")
        if removed:
            print(f"[SymbolRegistry] GC removed: {removed}")
        registry.print_status()

    elif args.gc:
        removed = registry.garbage_collect()
        if removed:
            registry.save()
            print(f"[SymbolRegistry] GC removed {len(removed)} symbol(s): {removed}")
        else:
            print("[SymbolRegistry] Nothing to collect.")


if __name__ == "__main__":
    main()
