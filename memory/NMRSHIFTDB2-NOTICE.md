# NMRShiftDB2 data notice

Contains information from nmrshiftdb2 ([www.nmrshiftdb.org](https://nmrshiftdb.nmr.uni-koeln.de)), which is made available here under the [nmrshiftdb2 Database License](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt).

A verbatim copy of that license is in [bench/NMRSHIFTDB2-DATABASE-LICENSE.txt](../bench/NMRSHIFTDB2-DATABASE-LICENSE.txt). This data is not covered by the repository's MIT License.

## What in this directory comes from nmrshiftdb2

| Path | Contents |
|---|---|
| `cases/` | Approved memory cases: for each, the nmrshiftdb2 record id and spectrum numbers, the structure (SMILES, scaffold), the formula, the solvent, the peak lists, and the atom-level environments taken from the database's atom assignments |
| `rejected/` | Cases that were proposed from nmrshiftdb2 records and rejected, with the reason |

Each case also holds the service's own output on that molecule (its proposed structures), which is not nmrshiftdb2 data. The derived database stays under the nmrshiftdb2 Database License; the machine-readable form is the JSON in `cases/`.
