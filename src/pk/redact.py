import regex

# A candidate number is the longest run of digits joined by single spaces or
# hyphens, judged whole: no rule ever looks at a sub-window of a run, so an
# order number cannot be redacted because some slice of it passes a check.
_NUMBER_RE = regex.compile(r"(?<!\d)(\+?)(\d(?:[ -]?\d)*)")
_AADHAAR_RE = regex.compile(r"[2-9]\d{11}|[2-9]\d{3}[ -]\d{4}[ -]\d{4}")
_PHONE_RE = regex.compile(r"(?:\+\d{1,3}[ -]?|0[ -]?)?[6-9]\d{4}[ -]?\d{5}")
_PAN_RE = regex.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")

LABELS = ("CARD", "AADHAAR", "PHONE", "PAN")


def placeholder(label):
    return f"[REDACTED-{label}]"


def luhn_ok(digits):
    total = 0
    for i, c in enumerate(reversed(digits)):
        d = int(c)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _judge_number(plus, run):
    digits = run.replace(" ", "").replace("-", "")
    if not plus:
        if 13 <= len(digits) <= 16 and luhn_ok(digits):
            return "CARD", False
        if _AADHAAR_RE.fullmatch(run):
            return "AADHAAR", False
    if _PHONE_RE.fullmatch(plus + run):
        return "PHONE", True
    return None, False


def find(text):
    """[(start, end, label)] of everything to redact, in text order."""
    spans = []
    for m in _NUMBER_RE.finditer(text):
        label, with_plus = _judge_number(m.group(1), m.group(2))
        if label:
            start = m.start(1) if with_plus else m.start(2)
            spans.append((start, m.end(2), label))
    for m in _PAN_RE.finditer(text):
        spans.append((m.start(), m.end(), "PAN"))
    spans.sort()
    return spans


def apply(text, spans):
    out = []
    pos = 0
    for start, end, label in spans:
        out.append(text[pos:start])
        out.append(placeholder(label))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def labels_of(spans):
    found = {label for _, _, label in spans}
    return [label for label in LABELS if label in found]


def redact(text):
    spans = find(text)
    return apply(text, spans), labels_of(spans)
