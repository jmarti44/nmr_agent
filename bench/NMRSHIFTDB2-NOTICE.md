# NMRShiftDB2 data notice

Contains information from nmrshiftdb2 ([www.nmrshiftdb.org](https://nmrshiftdb.nmr.uni-koeln.de)), which is made available here under the [nmrshiftdb2 Database License](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt).

A verbatim copy of that license is in [NMRSHIFTDB2-DATABASE-LICENSE.txt](NMRSHIFTDB2-DATABASE-LICENSE.txt) (downloaded 2026-11-14). This data is not covered by the repository's MIT License.

## What in this repository comes from nmrshiftdb2

| Path | Contents |
|---|---|
| `bench/results/`, `bench/results_main/`, `bench/results_unreported/` | Selected records (ids, structures, peak lists) and the service's answers on them |
| `bench/nmrshiftdb2_*_output.txt`, `bench/nmrshiftdb2_spotcheck_sample.txt`, `bench/verifier_nonbench_*_output.txt` | Statistics and samples computed from the database |
| `bench/report.md` | Results produced from the records above |
| `src/verify/shift_tables.py` | Shift ranges whose sources are listed as `NSDB2` or `NSDB2-P5` were derived from distributions computed on the database |

The database export itself (`nmrshiftdb2withsignals.sd`) is not included; it goes in `bench/data/`, which is not tracked.
