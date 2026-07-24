"""Grounded follow-up Q&A over ingested documents.

This service exists as much for what it proves as for what it does. It targets a
completely different schema (:class:`~app.schemas.analysis.ChatAnswer`), a
different prompt, and a dynamically built retrieval plan — and it drives the
*identical* :class:`~app.rag.pipeline.RAGPipeline` instance the resume analyzer
uses. That is US5 and SC-008 demonstrated in production code rather than in a
test fixture.

Conversation memory is a sliding window of the last N turns, not a running
summary. Summarising costs an extra generation per turn on a rate-limited free
tier, and it degrades exactly where it matters — a summary of "did they mention
Kubernetes?" loses the specific token the next retrieval needs.
"""

from __future__ import annotations

from collections import OrderedDict

from app.prompts.registry import PromptRegistry, schema_instructions
from app.rag.pipeline import RAGPipeline
from app.schemas.analysis import ChatAnswer, ChatTurn
from app.schemas.rag import (
    RetrievalPlan,
    RetrievalPlanStep,
    RetrievalTrace,
    StageTimings,
)
from app.utils.logging import get_logger
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


class ConversationMemory:
    """Bounded per-session sliding-window memory.

    Two independent bounds. ``max_turns`` caps context length per conversation;
    ``max_sessions`` caps total memory, evicting the least recently used session.
    Without the second bound this is a memory leak in any service that stays up
    longer than a demo.
    """

    def __init__(self, *, max_turns: int = 6, max_sessions: int = 64) -> None:
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[ChatTurn]] = OrderedDict()

    def history(self, session_id: str) -> list[ChatTurn]:
        """Return the retained turns for a session."""
        turns = self._sessions.get(session_id)
        if turns is None:
            return []
        self._sessions.move_to_end(session_id)
        return list(turns)

    def append(self, session_id: str, turn: ChatTurn) -> None:
        """Record a turn, trimming the window and evicting old sessions."""
        turns = self._sessions.setdefault(session_id, [])
        turns.append(turn)
        # Turns are stored in user/assistant pairs, so the window is trimmed in
        # pairs — leaving a dangling user message would show the model a question
        # with no answer and imply the previous answer was withheld.
        while len(turns) > self._max_turns * 2:
            del turns[0]
        self._sessions.move_to_end(session_id)

        while len(self._sessions) > self._max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
            logger.debug("evicted chat session", extra={"session_id": evicted})

    def clear(self, session_id: str | None = None) -> None:
        """Drop one session, or all of them."""
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    @staticmethod
    def render(turns: list[ChatTurn]) -> str:
        """Format turns for inclusion in a prompt."""
        if not turns:
            return "(no previous messages)"
        return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in turns)


class ChatResponse:
    """A chat answer with its retrieval provenance."""

    __slots__ = ("answer", "session_id", "timings", "trace")

    def __init__(
        self,
        answer: ChatAnswer,
        *,
        session_id: str,
        trace: RetrievalTrace,
        timings: StageTimings,
    ) -> None:
        self.answer = answer
        self.session_id = session_id
        self.trace = trace
        self.timings = timings


class ChatService:
    """Answers follow-up questions grounded in ingested documents.

    Args:
        pipeline: The same generic pipeline the analyzer uses.
        registry: Prompt template source.
        memory: Conversation memory.
        template_name: Chat prompt template.
        top_k: Passages retrieved per document.
    """

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        registry: PromptRegistry,
        memory: ConversationMemory | None = None,
        template_name: str = "chat_qa_v1",
        top_k: int = 4,
    ) -> None:
        self._pipeline = pipeline
        self._registry = registry
        self._memory = memory or ConversationMemory()
        self._template_name = template_name
        self._top_k = top_k

    def ask(
        self,
        *,
        session_id: str,
        message: str,
        document_ids: list[str],
        top_k: int | None = None,
    ) -> ChatResponse:
        """Answer a question using only passages from the named documents.

        Args:
            session_id: Conversation identifier.
            message: The user's question.
            document_ids: Documents the question may draw on.
            top_k: Override passages retrieved per document.

        Returns:
            The grounded answer with its retrieval trace.

        Raises:
            InsufficientContextError: Nothing relevant was retrieved.
            OutputValidationError: The repair budget was exhausted.
            LLMError: The generation provider failed.
        """
        timings = StageTimings()
        history = self._memory.history(session_id)

        with Stopwatch("total", timings):
            plan = self._build_plan(message, document_ids, top_k or self._top_k, history)
            spec = self._registry.get(self._template_name)

            result = self._pipeline.run(
                plan=plan,
                template=spec.compile(),
                schema=ChatAnswer,
                system=spec.system or None,
                timings=timings,
                output_schema=schema_instructions(ChatAnswer),
                history=ConversationMemory.render(history),
                question=message,
            )

        answer: ChatAnswer = result.value
        self._memory.append(session_id, ChatTurn(role="user", content=message))
        self._memory.append(session_id, ChatTurn(role="assistant", content=answer.answer))

        logger.info(
            "chat answered",
            extra={
                "session_id": session_id,
                "documents": len(document_ids),
                "citations": len(answer.citations),
                **timings.as_reported(),
            },
        )
        return ChatResponse(answer, session_id=session_id, trace=result.trace, timings=timings)

    def _build_plan(
        self, message: str, document_ids: list[str], top_k: int, history: list[ChatTurn]
    ) -> RetrievalPlan:
        """Build a one-step-per-document plan for this question."""
        query = self._rewrite_query(message, history)
        return RetrievalPlan(
            steps=tuple(
                RetrievalPlanStep(
                    name=f"document_{index + 1}",
                    query=query,
                    document_id=document_id,
                    top_k=top_k,
                )
                for index, document_id in enumerate(document_ids)
            )
        )

    @staticmethod
    def _rewrite_query(message: str, history: list[ChatTurn]) -> str:
        """Expand a follow-up question with recent context.

        Query rewriting without an extra model call. The problem it solves is
        concrete: "what about Kubernetes?" embeds to almost nothing useful on its
        own, because the subject of the question lives in the previous turn.
        Prefixing the last user message restores the missing terms for both the
        dense and the lexical retriever.

        The heuristic — rewrite only short questions — is deliberate. A long,
        self-contained question already carries its own context, and padding it
        with history would dilute the embedding rather than sharpen it.
        """
        if len(message.split()) > 8 or not history:
            return message
        previous = next(
            (turn.content for turn in reversed(history) if turn.role == "user"), ""
        )
        return f"{previous} {message}".strip() if previous else message
