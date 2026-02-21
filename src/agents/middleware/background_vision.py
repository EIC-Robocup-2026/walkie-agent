"""Middleware that injects live background-vision data into the system prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import SystemMessage

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

if TYPE_CHECKING:
    from src.vision.background_monitor import BackgroundVisionMonitor


class BackgroundVisionMiddleware(AgentMiddleware):
    """Appends the latest background vision snapshot to the system message.

    If the monitor has not yet completed a cycle (``latest_result`` is empty),
    nothing is injected so the agent is not misled by stale or absent data.

    Parameters
    ----------
    monitor:
        A :class:`~src.vision.background_monitor.BackgroundVisionMonitor` that
        has already been started via ``monitor.start()``.
    section_header:
        The Markdown header used for the injected section.
    """

    tools: list = []

    def __init__(
        self,
        monitor: "BackgroundVisionMonitor",
        section_header: str = "## Background Vision (live)",
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._header = section_header

    def _build_system_message(self, request: ModelRequest) -> SystemMessage | None:
        snapshot = self._monitor.latest_result
        if not snapshot:
            # No data yet – don't inject anything.
            return None

        extra_text = f"\n\n{self._header}\n{snapshot}"

        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": extra_text},
            ]
        else:
            new_content = [{"type": "text", "text": extra_text.lstrip()}]

        return SystemMessage(content=cast("list[str | dict[str, str]]", new_content))

    def wrap_model_call(self, request: ModelRequest, handler):
        new_sys = self._build_system_message(request)
        if new_sys is not None:
            request = request.override(system_message=new_sys)
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler):
        new_sys = self._build_system_message(request)
        if new_sys is not None:
            request = request.override(system_message=new_sys)
        return await handler(request)
