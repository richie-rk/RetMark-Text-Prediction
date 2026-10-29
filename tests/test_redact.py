import pytest

from pk.redact import luhn_ok, redact


@pytest.mark.parametrize(
    "text,expected,labels",
    [
        # Payment cards: 13-16 digits, single spaces or hyphens, Luhn.
        ("card 4111 1111 1111 1111 ok", "card [REDACTED-CARD] ok", ["CARD"]),
        ("card 4111-1111-1111-1111", "card [REDACTED-CARD]", ["CARD"]),
        ("card 4111111111111111.", "card [REDACTED-CARD].", ["CARD"]),
        ("amex 378282246310005", "amex [REDACTED-CARD]", ["CARD"]),
        ("card 4111 1111 1111 1112", "card 4111 1111 1111 1112", []),
        ("card 5105 1051 0510 5106", "card 5105 1051 0510 5106", []),
        ("order 123456789012 8 items", "order [REDACTED-CARD] items", ["CARD"]),
        # Aadhaar: 12 digits starting 2-9, contiguous or 4-4-4.
        ("id 234567890123", "id [REDACTED-AADHAAR]", ["AADHAAR"]),
        ("id 2345 6789 0123", "id [REDACTED-AADHAAR]", ["AADHAAR"]),
        ("id 2345-6789-0123", "id [REDACTED-AADHAAR]", ["AADHAAR"]),
        ("id 23456 7890123", "id 23456 7890123", []),
        # Phone: optional +CC or leading 0, 10 digits starting 6-9, 5+5 split.
        ("call +919876543210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call +91 98765 43210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call +91-98765-43210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call 09876543210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call 0 98765 43210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call 98765 43210 now", "call [REDACTED-PHONE] now", ["PHONE"]),
        ("call 9876543210", "call [REDACTED-PHONE]", ["PHONE"]),
        ("call 5876543210", "call 5876543210", []),
        # PAN: 5 letters, 4 digits, 1 letter, any case, word boundaries.
        ("pan ABCDE1234F.", "pan [REDACTED-PAN].", ["PAN"]),
        ("pan abcde1234f", "pan [REDACTED-PAN]", ["PAN"]),
        ("xabcde1234f", "xabcde1234f", []),
        # Negatives: order numbers starting with 1 are left alone.
        ("order 198765432101", "order 198765432101", []),
        ("order 123456789012", "order 123456789012", []),
        ("years 2019 2020 2021 2022", "years 2019 2020 2021 2022", []),
        ("pi is 3.14159", "pi is 3.14159", []),
        # Several kinds in one sentence keep rule order in labels.
        ("pan abcde1234f and 4111111111111111", "pan [REDACTED-PAN] and [REDACTED-CARD]", ["CARD", "PAN"]),
    ],
)
def test_redaction(text, expected, labels):
    out, got = redact(text)
    assert out == expected
    assert got == labels


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


def test_plus_never_starts_a_card_or_aadhaar():
    out, labels = redact("x +234567890123")
    assert "AADHAAR" not in labels
