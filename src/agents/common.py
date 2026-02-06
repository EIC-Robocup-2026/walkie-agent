from typing import Callable, cast
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, wrap_model_call


class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model = request.model.bind_tools(
            request.tools,
            parallel_tool_calls=False
        )
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model = request.model.bind_tools(
            request.tools,
            parallel_tool_calls=False
        )
        return await handler(request)


class InjectTodosIntoPromptMiddleware(AgentMiddleware):
    """Middleware that injects the current todos into the system prompt so the model can see them."""
    
    def _format_todos(self, todos: list) -> str:
        if not todos:
            return ""
        
        lines = ["\n\n## Current Todo List"]
        for i, todo in enumerate(todos, 1):
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
            }.get(todo.get("status", "pending"), "⬜")
            lines.append(f"{i}. {status_icon} [{todo.get('status', 'pending')}] {todo.get('content', '')}")
        
        return "\n".join(lines)
    
    def _inject_todos(self, request: ModelRequest) -> ModelRequest:
        todos = request.state.get("todos", [])
        if not todos:
            return request
        
        todos_text = self._format_todos(todos)
        
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": todos_text},
            ]
        else:
            new_system_content = [{"type": "text", "text": todos_text}]
        
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return request.override(system_message=new_system_message)
    
    def wrap_model_call(self, request, handler):
        request = self._inject_todos(request)
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request = self._inject_todos(request)
        return await handler(request)


@wrap_model_call
def sequential_tool_call_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    request.model = request.model.bind_tools(
        request.tools,
        parallel_tool_calls=False
    )
    return handler(request)