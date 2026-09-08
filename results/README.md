# Recorded runs

These files preserve the experiment record. The September 2026 cleanup moved them without changing their bytes or repeating training.
File paths inside JSON and logs describe the original machines and output locations.

| Folder | Record |
| --- | --- |
| `eight-seed/` | Final validation summary and full training log |
| `five-seed/` | Earlier training output and failure log |
| `smoke/` | Three one-epoch, one-batch pipeline checks |

Use the eight-seed summary for the reported validation result.
The five-seed logs preserve an earlier attempt. The smoke files check execution and do not support an accuracy claim.
Run new experiments under `runs/` so their outputs stay separate from these records.
