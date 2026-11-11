from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    location: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source: str
    location: str
    start: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    label: str
    chunk: Chunk


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    latency_ms: float = 0
    model_calls: int = 0


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Turn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
