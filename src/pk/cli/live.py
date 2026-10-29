import asyncio
import io
import uuid

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.filters import Condition
from rich.bar import Bar
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from pk.corrector import Corrector, HarperLS
from pk.logs import LogWriter, local_now, pending_clean, timestamp
from pk.persistence import InstanceLock, Workspace
from pk.session import LiveSession, OnlineTrainer
from pk.tokenizer import Tokenizer

RECORDING = ("fg:ansigreen bold", "● RECORDING")
PAUSED = ("fg:#ffaf00 bold", "❚❚ PAUSED — not learning")


def _word(s):
    return s.word if s is not None else ""


def toolbar_fragments(session):
    frags = []
    c = session.correction
    if c is not None:
        frags.append(("fg:ansired bold", f" {c.typed} → {c.suggestion} "))
        frags.append(("", "   "))
    slots = session.prediction.slots if session.prediction else None
    if slots is not None:
        frags.append(("", f"{_word(slots.second)}  "))
        frags.append(("bold reverse", f"[{_word(slots.primary)}]"))
        frags.append(("", f"  {_word(slots.third)}   |  graph: {_word(slots.side)}"))
    frags.append(("", "\n"))
    engine, graph = session.handover.rates()
    n, window = len(session.handover.records), session.handover.window
    frags.append(("bold", f"{session.mode}"))
    frags.append(("", f"  engine {engine:.0%} · graph {graph:.0%}  ({n}/{window} words)   "))
    frags.append(PAUSED if session.paused else RECORDING)
    if session.error is not None:
        frags.append(("fg:ansired", f"   {session.error}"))
    trainer = session.trainer
    if trainer is not None and trainer.error is not None:
        frags.append(("fg:ansired", f"   online update failed: {trainer.error}"))
    return frags


def render_inspect(info, width):
    console = Console(file=io.StringIO(), width=width, force_terminal=True, color_system="truecolor")
    tokens = Text(" ".join(f"{t!r}:{i}" for t, i in info["tokens"]), overflow="fold")
    norms = Table.grid(padding=(0, 1))
    scale = max(info["S_s_norm"], info["S_f_norm"], info["z_s_norm"], info["z_f_norm"], 1e-9)
    for name in ("S_s_norm", "S_f_norm", "z_s_norm", "z_f_norm"):
        label = {"S_s_norm": "‖S slow‖", "S_f_norm": "‖S fast‖", "z_s_norm": "‖z slow‖", "z_f_norm": "‖z fast‖"}[name]
        norms.add_row(label, f"{info[name]:9.4f}", Bar(size=scale, begin=0, end=info[name], width=24))
    top = Table("next token", "p", "", box=None, padding=(0, 1))
    for t, p in info["top"]:
        top.add_row(repr(t), f"{p:.4f}", Bar(size=1.0, begin=0, end=p, width=24))
    mean = "—" if info["mlp_mean"] is None else f"{info['mlp_mean']:+.4f}"
    mlp = Text(f"MLP hidden: mean {mean}, above zero {info['mlp_frac_pos']:.1%}")
    marker = Text("◆ boundary decay fired", style="bold magenta") if info["boundary_fired"] else Text("")
    grid = Table.grid(padding=(0, 4))
    grid.add_row(norms, top)
    console.print(Group(Text("committed tokens", style="bold"), tokens, grid, mlp, marker))
    return console.file.getvalue()


class Harness:
    def __init__(self, session, cfg, ws, inspect, banner):
        self.session = session
        self.cfg = cfg
        self.ws = ws
        self.inspect = inspect
        self.banner = banner
        self.toolbar = []
        self.panel = ANSI("")
        self._debounce = None
        self._writing = False
        self.app = self._build()
        self.refresh()

    def _body(self):
        frags = []
        if self.banner:
            frags.append(("fg:#ffaf00", self.banner + "\n\n"))
        # A literal tab would be drawn wider than one cell and push the
        # cursor out of step with the text.
        frags.append(("", self.session.text.replace("\t", "→")))
        return frags

    def _cursor(self):
        before = self.session.text[: self.session.cursor]
        offset = self.banner.count("\n") + 2 if self.banner else 0
        return Point(x=len(before) - (before.rfind("\n") + 1), y=before.count("\n") + offset)

    def _build(self):
        kb = KeyBindings()
        s = self.session

        def act(fn):
            def handler(event):
                fn()
                self.changed()

            return handler

        @kb.add("<any>")
        def _(event):
            if event.data and event.data.isprintable():
                s.type(event.data)
                self.changed()

        kb.add("backspace")(act(s.backspace))
        kb.add("enter")(act(s.enter))
        kb.add("tab")(act(s.tab))
        kb.add("insert")(act(s.insert))
        kb.add("f2")(act(s.f2))
        kb.add("left")(act(s.left))
        kb.add("right")(act(s.right))
        kb.add("c-p")(act(s.toggle_pause))

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event):
            event.app.exit()

        body = Window(
            FormattedTextControl(self._body, get_cursor_position=self._cursor, show_cursor=True, focusable=True),
            wrap_lines=True,
        )
        panel = ConditionalContainer(
            Window(FormattedTextControl(lambda: self.panel), height=16), filter=Condition(lambda: self.inspect)
        )
        bar = Window(FormattedTextControl(lambda: self.toolbar), height=2, style="bg:#1c1c1c")
        return Application(layout=Layout(HSplit([body, panel, bar])), key_bindings=kb, full_screen=True)

    def changed(self):
        """Echo the keystroke at once; redraw the toolbar after debounce_ms quiet."""
        loop = asyncio.get_event_loop()
        if self._debounce is not None:
            self._debounce.cancel()
        self._debounce = loop.call_later(self.cfg.predict.debounce_ms / 1000, self.refresh)

    def refresh(self):
        self.toolbar = toolbar_fragments(self.session)
        if self.inspect:
            width = 120
            try:
                width = self.app.output.get_size().columns
            except Exception:
                pass
            self.panel = ANSI(render_inspect(self.session.inspect(), width))
        try:
            self.app.invalidate()
        except Exception:
            pass

    async def _background(self):
        loop = asyncio.get_event_loop()
        interval = self.cfg.predict.debounce_ms / 1000
        last_checkpoint = loop.time()
        while True:
            await asyncio.sleep(interval)
            if self.session.poll():
                self.refresh()
            if loop.time() - last_checkpoint >= self.cfg.persist.checkpoint_interval_s and not self._writing:
                last_checkpoint = loop.time()
                self.ws.state.handover = self.session.handover.to_list()
                write = self.ws.capture(timestamp(local_now()), self.session.trainer.lock)
                self._writing = True
                loop.run_in_executor(None, write).add_done_callback(lambda _: setattr(self, "_writing", False))

    def run(self):
        async def main():
            task = asyncio.ensure_future(self._background())
            try:
                await self.app.run_async()
            finally:
                task.cancel()

        asyncio.run(main())


def run_live(cfg, inspect=False, echo=print):
    """Run the harness. Returns the process exit code."""
    lock = InstanceLock(cfg.paths.data_dir)
    lock.acquire()
    harper = writer = trainer = session = None
    ws = None
    completed = False
    try:
        tok = Tokenizer(cfg.model.boundary_chars)
        ws = Workspace.load(cfg, tok.vocab_size)
        harper = HarperLS(cfg.correct.harper_ls_path)
        corrector = Corrector(harper, cfg.correct.dialect, ws.dictionary)
        corrector.warm()
        writer = LogWriter(cfg.paths.data_dir, cfg.logs.flush_interval_s, str(uuid.uuid4()))
        trainer = OnlineTrainer(ws, tok, cfg)
        session = LiveSession(cfg, tok, ws, corrector, writer, trainer, async_checks=True)
        pending = len(pending_clean(cfg.paths.data_dir, ws.state.offsets))
        banner = f"Pending consolidation: {pending} sentences. Run pk consolidate after this session." if pending else ""
        Harness(session, cfg, ws, inspect, banner).run()
        completed = True
        return 0
    finally:
        # Ctrl+C and errors land here too: everything typed is kept, and each
        # step runs even if an earlier one fails, so the lock is always freed.
        errors = []

        def attempt(fn):
            try:
                fn()
            except Exception as e:
                errors.append(e)

        if session is not None:
            attempt(session.close)
        if trainer is not None:
            attempt(trainer.close)
        if writer is not None:
            attempt(writer.close)
        if ws is not None and session is not None:
            attempt(lambda: ws.checkpoint(timestamp(local_now()), trainer.lock))
        if harper is not None:
            attempt(harper.close)
        lock.release()
        if errors and completed:
            raise errors[0]
