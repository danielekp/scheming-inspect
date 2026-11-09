"""Fixed questions and blind interrogation of recorded model actions.

Sources, sample construction, solvers, scorers, tasks, and the CLI live in
separate modules. Public entry points are also available from this package.
"""

from .cli import (
    main,
)
from .samples import (
    COPY,
    DELETION,
    OVERSIGHT,
    PROMPT,
    REPLACEMENT,
    Action,
    eligible_actions,
    fixed_questions,
    fixed_sample,
    fixed_sample_id,
    fixed_samples,
    interrogation_sample,
    interrogation_sample_id,
    interrogation_samples,
    matched_sources,
    source_metadata,
)
from .scorers import (
    admission_metrics,
    interrogation_verdict,
    response_failure_score,
)
from .solvers import (
    IncompleteTargetResponse,
    InvalidInterrogatorResponse,
    blind_interrogator_messages,
    fixed_followup_solver,
    interrogation_solver,
    record_response_failure,
    target_reply,
)
from .sources import (
    MATCH_KEYS,
    SourceCase,
    boolean_flags,
    load_sources,
    matching_context,
    read_behavior,
    source_exclusion_reason,
    source_from_sample,
    source_key,
    task_description,
    validate_transcript,
)
from .tasks import (
    build_task,
    task_from_samples,
)

__all__ = [
    "MATCH_KEYS",
    "SourceCase",
    "validate_transcript",
    "task_description",
    "read_behavior",
    "boolean_flags",
    "source_key",
    "matching_context",
    "source_exclusion_reason",
    "source_from_sample",
    "load_sources",
    "Action",
    "OVERSIGHT",
    "REPLACEMENT",
    "COPY",
    "DELETION",
    "PROMPT",
    "eligible_actions",
    "fixed_questions",
    "source_metadata",
    "fixed_sample_id",
    "fixed_sample",
    "interrogation_sample_id",
    "interrogation_sample",
    "fixed_samples",
    "matched_sources",
    "interrogation_samples",
    "InvalidInterrogatorResponse",
    "IncompleteTargetResponse",
    "record_response_failure",
    "target_reply",
    "fixed_followup_solver",
    "blind_interrogator_messages",
    "interrogation_solver",
    "response_failure_score",
    "admission_metrics",
    "interrogation_verdict",
    "task_from_samples",
    "build_task",
    "main",
]
