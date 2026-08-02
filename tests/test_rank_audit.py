import numpy as np

from orthogmm.diagnostics.rank_audit import RankAudit


def test_rank_audit_returns_requested_rows() -> None:
    rng = np.random.default_rng(123)
    n = 50
    qg = 6
    qh = 3
    p = 2

    g = rng.normal(size=(n, qg))
    h = rng.normal(size=(n, qh))
    G = rng.normal(size=(qg, p))
    H = rng.normal(size=(qh, p))

    rows = RankAudit(
        minimum_rank=1,
        maximum_rank=4,
        ridge=1e-10,
    ).run(g, h, G, H)

    assert [row.rank for row in rows] == [1, 2, 3, 4]
    assert all(row.empirical_rank == 6 for row in rows)
    assert all(row.raw_rank_omega_gg == row.rank for row in rows)


def test_rank_audit_orthogonality_is_small() -> None:
    rng = np.random.default_rng(456)
    n = 100
    qg = 5
    qh = 2
    p = 2

    g = rng.normal(size=(n, qg))
    h = g[:, :2] + 0.1 * rng.normal(size=(n, qh))
    G = rng.normal(size=(qg, p))
    H = rng.normal(size=(qh, p))

    rows = RankAudit(
        minimum_rank=5,
        maximum_rank=5,
    ).run(g, h, G, H)

    assert rows[0].orthogonality_norm < 1e-10
