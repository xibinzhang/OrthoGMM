"""Generate Section 5 tables and figures from authoritative CSV files."""

from pathlib import Path

from orthogmm.reporting import generate_section5_outputs


def main() -> None:
    created = generate_section5_outputs(
        baseline_csv="results/section5_baseline_R200.csv",
        basis_csv="results/section5_basis_R200.csv",
        quadrature_csv="results/section5_quadrature_R200.csv",
        output_dir=Path("results/section5_report"),
    )

    print("Created:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
