"""Shared example: aspirin in DMSO-d6 (nine analyte 13C signals, SPEC §9.2) plus residual-solvent peaks."""

from src.schemas import AnalysisRequest

ASPIRIN_DMSO = {
    "input_format": "peaklist",
    "context": {
        "molecular_formula": "C9H8O4",
        "solvent": "DMSO-d6",
        "spectrometer_mhz": 400.13,
        "notes": "white crystalline solid",
    },
    "max_candidates": 3,
    "peaks": {
        "peaks": [
            {"id": 1, "nucleus": "1H", "shift_ppm": 12.95, "multiplicity": "br", "integral": 1.0},
            {
                "id": 2,
                "nucleus": "1H",
                "shift_ppm": 7.95,
                "multiplicity": "dd",
                "coupling_hz": [7.8, 1.6],
                "integral": 1.0,
            },
            {
                "id": 3,
                "nucleus": "1H",
                "shift_ppm": 7.64,
                "multiplicity": "td",
                "coupling_hz": [7.8, 1.6],
                "integral": 1.0,
            },
            {
                "id": 4,
                "nucleus": "1H",
                "shift_ppm": 7.37,
                "multiplicity": "td",
                "coupling_hz": [7.8, 1.2],
                "integral": 1.0,
            },
            {
                "id": 5,
                "nucleus": "1H",
                "shift_ppm": 7.17,
                "multiplicity": "dd",
                "coupling_hz": [8.0, 1.2],
                "integral": 1.0,
            },
            {"id": 6, "nucleus": "1H", "shift_ppm": 2.50, "multiplicity": "quint", "integral": 0.1},
            {"id": 7, "nucleus": "1H", "shift_ppm": 2.24, "multiplicity": "s", "integral": 3.0},
            {"id": 8, "nucleus": "13C", "shift_ppm": 169.1},
            {"id": 9, "nucleus": "13C", "shift_ppm": 166.3},
            {"id": 10, "nucleus": "13C", "shift_ppm": 150.3},
            {"id": 11, "nucleus": "13C", "shift_ppm": 134.2},
            {"id": 12, "nucleus": "13C", "shift_ppm": 131.2},
            {"id": 13, "nucleus": "13C", "shift_ppm": 126.1},
            {"id": 14, "nucleus": "13C", "shift_ppm": 124.0},
            {"id": 15, "nucleus": "13C", "shift_ppm": 123.8},
            {"id": 16, "nucleus": "13C", "shift_ppm": 39.5},
            {"id": 17, "nucleus": "13C", "shift_ppm": 20.8},
        ]
    },
}


def aspirin_request(**overrides) -> AnalysisRequest:
    return AnalysisRequest.model_validate({**ASPIRIN_DMSO, **overrides})
