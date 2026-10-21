"""Fusão de rankings por RRF (não soma scores de modelos distintos)."""


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: listas de chaves ordenadas por relevância. Retorna (chave, score RRF)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
