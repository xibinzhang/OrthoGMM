"""Construct the canonical PyBLP Petrin benchmark without estimation."""

from orthogmm.model.petrin import build_petrin_problem


def main() -> None:
    setup = build_petrin_problem()

    print("Petrin PyBLP benchmark")
    print(f"Markets: {setup.n_markets}")
    print(f"Products: {setup.n_products}")
    print(f"Agents: {setup.n_agents}")
    print(f"Micro moments: {len(setup.micro_moments)}")
    print(f"Initial sigma shape: {setup.initial_sigma.shape}")
    print(f"Initial pi shape: {setup.initial_pi.shape}")
    print("Market IDs:")
    print(setup.market_ids)


if __name__ == "__main__":
    main()
