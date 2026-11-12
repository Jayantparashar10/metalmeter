"""Neural Engine clients: which processes have an open connection to the ANE.

macOS has no busy counter for the Neural Engine. Its user clients carry only creator,
entitlement and locking properties, so a connection says nothing about whether the
process is using the ANE at that moment.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .. import parse
from .._bind import iokit
from . import never_raises

# On an M2 with macOS 27 every client is under H1xANELoadBalancer. H11ANEIn is the driver
# expected on earlier chips; that case has not been checked on hardware yet.
PROVIDERS = ("H1xANELoadBalancer", "H11ANEIn")


@never_raises
def read(pids: Optional[Iterable[int]] = None) -> Optional[dict[str, Any]]:
    """Read the Neural Engine's user clients.

    Args:
        pids: if given, only these processes' clients are returned.

    Returns:
        ``{"providers": [...], "clients": [{"provider", "creator"}]}``, or None if no ANE
        driver exists (Intel Macs, virtual machines).
    """
    if not iokit.AVAILABLE:
        return None
    want = None if pids is None else set(pids)
    providers, clients = [], []
    for provider in PROVIDERS:
        with iokit.matching(provider) as services:
            if services:
                providers.append(provider)
            for svc in services:
                with iokit.children(svc) as (kids, _):
                    for k in kids:
                        creator = iokit.prop(k, "IOUserClientCreator")
                        if creator is None:
                            continue
                        if want is not None:
                            p = parse.parse_creator(creator)
                            if p is None or p[0] not in want:
                                continue
                        clients.append({"provider": provider, "creator": creator})
    if not providers:
        return None
    return {"providers": providers, "clients": clients}
