"""Where an ecology failure sits, and how a provider's failure becomes one.

:class:`EcologyError` is the root of everything the ecology raises, and it
derives from the platform's existing :class:`~liq.evolution.errors.LiqEvolutionError`
rather than starting a hierarchy of its own. A second independent root would
force every caller to know which of two families a failure came from, which is
the opposite of what a root is for.

Leaf types are not catalogued here. Each arrives with the code that raises it,
so a type never outlives the behaviour that justified it and no caller can
catch something nothing throws.

**The translation convention.** A provider's exception is a fact about a
library, not about the ecology, so it never travels inward. At the boundary an
adapter catches what its provider raises and re-raises it as the
:class:`EcologyError` subtype its own slice defines, with the original attached
as ``__cause__``. Three rules make that safe, and :func:`translating` applies
all three:

* The original is preserved as ``__cause__``, so the traceback still shows what
  actually broke. Nothing is swallowed.
* The ecology error's message carries the provider exception's *type name* and
  a description of what was being attempted — never the provider's message
  text. Provider messages routinely quote URLs, connection strings, request
  bodies and tokens; copying one into a new message is how a credential reaches
  a log line that was reviewed and thought safe.
* An :class:`EcologyError` raised inside the block travels out untouched, so a
  failure is described once rather than wrapped at every level it passes.

Domain and use-case code catches :class:`EcologyError` or one of its subtypes.
It never names a provider's exception type, because doing so would put the
provider's vocabulary in a signature the port was written to keep it out of.

Argument and contract violations inside the ecology are a different thing and
raise the stdlib exception that fits — a caller in this codebase passing an
impossible value is a defect to fix, not a condition to handle, and dressing it
as a domain error invites someone to catch it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from liq.evolution.errors import LiqEvolutionError

__all__ = [
    "EcologyError",
    "translating",
]


class EcologyError(LiqEvolutionError):
    """Root of every error the ecology raises.

    Catching this catches everything that originates inside the ecology,
    including errors an adapter translated from its provider, and nothing that
    originates elsewhere in the platform.
    """


@contextmanager
def translating(
    provider_errors: type[Exception] | tuple[type[Exception], ...],
    *,
    into: type[EcologyError],
    action: str,
) -> Iterator[None]:
    """Re-raise a provider's failure as an ecology error, keeping the original.

    Args:
        provider_errors: The exception type, or types, the provider raises for
            the failure being translated. Naming them is deliberate: a blanket
            catch would translate defects in the adapter's own code as if the
            provider had failed.
        into: The :class:`EcologyError` subtype the slice using this adapter
            defines for the failure. There is no default, because the root is a
            thing to catch rather than a thing to raise.
        action: A short, static description of what was being attempted, such
            as ``"reading the next decision point"``. It is written in the
            code, so it cannot carry a value, a payload or a credential.

    Raises:
        EcologyError: Of type ``into``, whenever the block raises one of
            ``provider_errors``. The original is attached as ``__cause__``.

    Example:
        An adapter whose slice defines ``WindowUnavailable(EcologyError)`` and
        whose provider raises ``TimeoutError`` writes::

            with translating(
                TimeoutError,
                into=WindowUnavailable,
                action="reading the next decision point",
            ):
                ...

        A timeout then leaves the boundary as ``WindowUnavailable`` reading
        ``TimeoutError while reading the next decision point``, with the
        original timeout — and whatever it had to say about hosts and
        credentials — reachable only as ``__cause__``.
    """
    try:
        yield
    except EcologyError:
        raise
    except provider_errors as exc:
        raise into(f"{type(exc).__name__} while {action}") from exc
