import csv
from pathlib import Path

import numpy as np
import pytest

from examples.petrin_localized_sop_pipeline import load_localized_theta


def test_load_localized_theta(tmp_path: Path) -> None:
    path = tmp_path / "localization.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["parameter", "initial", "localized", "change"],
        )
        writer.writeheader()
        for index in range(13):
            writer.writerow(
                {
                    "parameter": f"p{index}",
                    "initial": 0.0,
                    "localized": float(index),
                    "change": float(index),
                }
            )

    names, theta = load_localized_theta(path)

    assert names[0] == "p0"
    assert names[-1] == "p12"
    np.testing.assert_allclose(theta, np.arange(13.0))


def test_load_localized_theta_requires_13_rows(tmp_path: Path) -> None:
    path = tmp_path / "localization.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["parameter", "localized"],
        )
        writer.writeheader()
        writer.writerow({"parameter": "p0", "localized": 1.0})

    with pytest.raises(ValueError, match="Expected 13"):
        load_localized_theta(path)
