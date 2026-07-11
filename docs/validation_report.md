# Data Validation Report

Generated: 2026-07-10 08:07

## Hard gates

- ✅ panel has 20,774 rows (>1,000 expected)
- ✅ no null supplier names
- ✅ no duplicate supplier-quarters (found 0)
- ✅ mill_produced_MT: no negative quantities
- ✅ mill_dispatched_MT: no negative quantities
- ✅ vaighai_offtake_est_MT: no negative quantities
- ✅ vaighai_purchased_MT: no negative quantities
- ✅ fiscal_quarter values all valid
- ✅ weather actuals present, non-negative

## Warnings / observations

- ✅ UNKNOWN-region rows: 96 (0.5%)
- ✅ share>100% rows after capping: 0
- ✅ MIR-only supplier-quarters (estimate vs actual gap): 16,771
