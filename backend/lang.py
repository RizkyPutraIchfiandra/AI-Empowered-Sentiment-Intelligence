"""Language detection for routing between the English and Indonesian models.

A stop word vote rather than a library: the two languages share almost no
function words, so counting which set appears more often is accurate enough to
pick a model, and it adds no dependency.
"""

import re

# Function words only. Content words would overlap too much with loanwords.
ID_MARKERS = frozenset(
    """
    yang dan di ke dari ini itu tidak dengan untuk ada saya aku kamu kami kita
    adalah akan sudah belum bisa juga pada dalam saja kalau karena lebih sangat
    banget nya lah dong deh sih aja gak nggak ga udah bgt sama oleh atau tapi
    tetapi jadi bikin buat punya mau pengen harus masih cuma hanya biar supaya
    gimana kenapa dimana siapa apa mereka dia nih tuh kok ya iya
    """.split()
)

EN_MARKERS = frozenset(
    """
    the and is are was were to of in on at for with this that these those it
    not no but or so if because very really just too also have has had do does
    did can could would should will my your our their they he she we you i
    there here what when where why how about from than then been being am
    """.split()
)

TOKEN_RE = re.compile(r"[a-z']+")


def detect(text: str) -> str:
    """Return 'id', 'en', or 'unknown' when neither language is in evidence."""
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return "unknown"

    id_hits = sum(t in ID_MARKERS for t in tokens)
    en_hits = sum(t in EN_MARKERS for t in tokens)

    # Require at least one marker and a clear winner. A tie on short input is
    # not enough to pick a model, and guessing there is how you get a confident
    # answer about a sentence the model never saw.
    if id_hits == en_hits:
        return "unknown"
    return "id" if id_hits > en_hits else "en"


if __name__ == "__main__":
    cases = [
        ("Hai sayang, kamu udah makan? aku sayang banget sama kamu.", "id"),
        ("The build quality is outstanding and it arrived two days early.", "en"),
        ("pelayanan nya lambat dan makanan nya dingin", "id"),
        ("not good at all, waste of money", "en"),
        ("asdf qwer zxcv", "unknown"),
        ("", "unknown"),
    ]
    for text, expected in cases:
        got = detect(text)
        assert got == expected, f"{text!r}: expected {expected}, got {got}"
    print(f"lang self-check passed ({len(cases)} cases)")
