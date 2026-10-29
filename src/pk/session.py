from pk.logs import SentenceEvents


class Recorder:
    """Raw and clean log streams for one harness run, gated by pause.

    A pause toggle ends the current sentence: whatever was recorded before
    the toggle is redacted and queued but never learned, and nothing typed
    while paused reaches a log or a learner.
    """

    def __init__(self, writer, paused):
        self.writer = writer
        self.paused = paused
        self._events = SentenceEvents()

    def event(self, type, **fields):
        if not self.paused:
            self._events.add(self.writer.record(type, **fields))

    def key(self, char):
        self.event("key", key=char)

    def special(self, name):
        self.event("special", key=name)

    def observe(self, sentence_text):
        if not self.paused:
            self._events.observe(sentence_text)

    def complete_sentence(self, raw_text, clean_text):
        """Close the current sentence. Returns True if it is to be learned."""
        self._events.finish(raw_text, self.writer)
        if self.paused:
            return False
        if clean_text is not None:
            self.writer.queue("clean", self.writer.clean_record(clean_text))
        return True

    def toggle_pause(self, unfinished_text):
        if not self.paused:
            self._events.finish(unfinished_text, self.writer)
            self.writer.queue("raw", self.writer.record("pause"))
            self.paused = True
        else:
            self._events = SentenceEvents()
            self.paused = False
            self.writer.queue("raw", self.writer.record("resume"))
        return self.paused

    def close(self, unfinished_text):
        self._events.finish(unfinished_text, self.writer)
