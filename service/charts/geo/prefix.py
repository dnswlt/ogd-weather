from bisect import bisect_left, bisect_right
from typing import Any, Callable, Iterable
import unicodedata


_REPLACEMENTS = {
    ord("ß"): "ss",
    ord("ẞ"): "SS",
    ord("æ"): "ae",
    ord("Æ"): "Ae",
    ord("œ"): "oe",
    ord("Œ"): "Oe",
    ord("ø"): "o",
    ord("Ø"): "O",
    ord("å"): "a",
    ord("Å"): "A",
    ord("ð"): "d",
    ord("Ð"): "D",
    ord("þ"): "th",
    ord("Þ"): "Th",
    ord("ł"): "l",
    ord("Ł"): "L",
    ord("đ"): "d",
    ord("Đ"): "D",
    ord("ğ"): "g",
    ord("Ğ"): "G",
    ord("ħ"): "h",
    ord("Ħ"): "H",
    ord("ı"): "i",
    ord("İ"): "I",
    ord("’"): "'",
    ord("‘"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("–"): "-",
    ord("—"): "-",
    ord("·"): "-",
    ord("‧"): "-",
    ord("\xa0"): " ",
}


def _ascii_fold(text: str, unknown: str = "?") -> str:
    """ASCII-only version of `text` with diacritics dropped and special letters transliterated."""
    t = text.translate(_REPLACEMENTS)  # special-cases (ß→ss, etc.)
    t = unicodedata.normalize("NFKD", t)  # split base + combining marks
    out = []
    for ch in t:
        code = ord(ch)
        if code < 128:
            out.append(ch)  # keep ASCII
        else:
            if unicodedata.category(ch) == "Mn":  # drop combining diacritics
                continue
            out.append(unknown)  # anything else: '?'
    return "".join(out)


class PrefixLookup:
    """
    ASCII-folded, case-insensitive prefix lookup over a corpus of strings.
    Uses binary search over a sorted, normalized index.
    """

    def __init__(self, corpus: Iterable[Any], key: Callable[[Any], str] = str) -> None:
        # Build (normalized, original) pairs; de-duplicate originals
        ts = []
        for item in corpus:
            if item is None:
                continue
            k = key(item)
            n = _ascii_fold(k).casefold()
            ts.append((n, k, item))

        # sort by normalized, then original (tuple sort)
        ts.sort(key=lambda t: (t[0], t[1]))

        self._norms = [n for n, _, _ in ts]
        self._items = [t for _, _, t in ts]

    def find_prefix(self, query: str, limit: int = 10) -> list[Any]:
        """
        Return up to `limit` terms from the corpus whose normalized form starts with the
        normalized `query` prefix. Empty query -> [].
        """
        if not query:
            return []

        q = _ascii_fold(query).casefold()
        # Find the slice of items with prefix q
        lo = bisect_left(self._norms, q)
        hi = bisect_right(self._norms, q + "\uffff")

        # Trim to limit without scanning the whole slice
        if limit < 0:
            limit = 0
        return self._items[lo : min(hi, lo + limit)]
