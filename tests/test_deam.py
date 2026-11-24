"""DEAM genres: first listed genre, case-insensitive; the rest -> OTHER (not 7)."""
from eval import deam


def test_genre_mapping():
    assert deam._genre("\tRock\t") == deam.GENRES.index("Rock")
    assert deam._genre("SoulRB-Country-Folk") == deam.OTHER
    assert deam._genre("Country-Folk") == deam.GENRES.index("Country")
    assert deam._genre("jazz") == deam.GENRES.index("Jazz")
    assert deam.OTHER != 7          # 7 means "unseen" in the shared analysis


def test_split_is_disjoint_and_fixed():
    a, b = deam.splits(1000), deam.splits(1000)
    for x, y in zip(a, b):
        assert (x == y).all()
    allidx = set()
    for part in a:
        assert not allidx & set(part.tolist())
        allidx |= set(part.tolist())
    assert len(allidx) == 1000
