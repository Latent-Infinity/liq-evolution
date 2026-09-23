"""What one pass over history is configured by — and what cannot be configured.

A configuration schema is read twice: once by whoever fills it in, and once by
whoever later asks what a run could possibly have done. The second reading is
the reason this schema is small. Every key here is a knob a run can be turned
by, so a key that exists is a behaviour someone can ask for, whether or not
anybody has.

**There is no epoch, replay, rewind or pass count, and its absence is a
property rather than an oversight.** History is walked once, forward. A key
offering a second pass would not merely permit a mistake — it would make the
mistake invisible, because a figure computed over history counted twice looks
exactly like a figure computed over twice as much history, and every number
downstream of it inherits the repetition silently. The claim that no such key
exists is asserted mechanically against this schema, so adding one is a visible
change to what the run is allowed to be rather than a line in a file.

**Continuity across a split boundary is stated, not assumed.** A walk-forward
boundary divides what history is *used for*; it is not an event in an agent's
life. An agent alive on either side of one is the same agent with the same
learned state, which is what makes the boundary a test of generalisation rather
than a restart in disguise. The alternative reading — that everything begins
again at a fold — is not implemented, so asking for it is refused here instead
of being quietly ignored: a setting that is read, disregarded and recorded in
provenance is worse than one that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "EcologyConfig",
]


@dataclass(frozen=True)
class EcologyConfig:
    """How one pass over history is driven.

    Attributes:
        run_id: Identity the pass is recorded under. Required, because a result
            that cannot be named cannot be reproduced or compared.
        window_carries_across_segments: Whether an agent's view of history
            survives a walk-forward split boundary. Continuous is the only
            reading implemented; it is carried here so a run's provenance
            records which reading it ran under rather than leaving a reader to
            infer it from the code of the day.

    Raises:
        ValueError: If the pass could not be identified, or asks for a
            discontinuity at split boundaries that nothing implements.
    """

    run_id: str
    window_carries_across_segments: bool = True

    def __post_init__(self) -> None:
        """Refuse a pass that cannot be named or asks for what does not exist."""
        if not self.run_id:
            raise ValueError(
                "a run must carry an identity: a result nothing names cannot be "
                "reproduced, compared or withdrawn"
            )
        if not self.window_carries_across_segments:
            raise ValueError(
                "window_carries_across_segments=False asks for an agent's view of "
                "history to restart at a walk-forward split boundary; nothing "
                "implements that, and accepting the setting would record a policy "
                "in provenance that the run did not follow"
            )
