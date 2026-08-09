# Troubleshooting

## When the round number is greater than the number of minimal reference states + 1

*(applies to `rsr.run_ref_extraction_by_mcs` with `min_ref_search=True`, the default)*

When `min_ref_search=True`, each round finds a minimal reference state that cannot be overridden by a reference state that is found in subsequent rounds. In a **coherent** system, the number of rounds
should therefore be the same as (the number of distinct minimal reference states
found) + 1.

If the round number is greater than *(number of minimal reference
states) + 1*, this indicates the likelihood that your system function is **not
coherent** — i.e. improving a component's state does not consistently improve
(or leave unchanged) the system state.
