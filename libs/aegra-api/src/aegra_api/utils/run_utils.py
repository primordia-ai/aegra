import copy
from typing import Any


def sanitize_for_db(obj: Any) -> Any:
    """Recursively remove NUL bytes (\\u0000) and unpaired surrogates from objects.

    PostgreSQL jsonb and text columns do not support NUL bytes in strings and
    will raise UntranslatableCharacter errors. Unpaired surrogates also cause
    encoding errors when sending data to the database.
    """
    if isinstance(obj, str):
        # Remove NUL bytes and strip unpaired surrogates by re-encoding to UTF-8
        return obj.replace("\u0000", "").encode("utf-8", "ignore").decode("utf-8")
    elif isinstance(obj, bytes):
        # Decode bytes to string, ignoring errors and removing NUL bytes
        try:
            return obj.decode("utf-8", "ignore").replace("\u0000", "")
        except Exception:
            return str(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_for_db(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_db(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_db(v) for v in obj)
    return obj


import structlog

logger = structlog.getLogger(__name__)


def _should_skip_event(raw_event: Any) -> bool:
    """Check if an event should be skipped based on langsmith:nostream tag"""
    try:
        # Check if the event has metadata with tags containing 'langsmith:nostream'
        if isinstance(raw_event, tuple) and len(raw_event) >= 2:
            # For tuple events, check the third element (metadata tuple)
            metadata_tuple = raw_event[len(raw_event) - 1]
            if isinstance(metadata_tuple, tuple) and len(metadata_tuple) >= 2:
                # Get the second item in the metadata tuple
                metadata = metadata_tuple[1]
                if isinstance(metadata, dict) and "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list) and "langsmith:nostream" in tags:
                        return True
        return False
    except Exception:
        # If we can't parse the event structure, don't skip it
        return False


def _merge_jsonb(*objects: dict) -> dict:
    """Mimics PostgreSQL's JSONB merge behavior"""
    result = {}
    for obj in objects:
        if obj is not None:
            result.update(copy.deepcopy(obj))
    return result


async def _filter_context_by_schema(context: dict[str, Any], context_schema: dict | None) -> dict[str, Any]:
    """Filter context parameters based on the context schema."""
    if not context_schema or not context:
        return context

    # Extract valid properties from the schema
    properties = context_schema.get("properties", {})
    if not properties:
        return context

    # Filter context to only include parameters defined in the schema
    filtered_context = {}
    for key, value in context.items():
        if key in properties:
            filtered_context[key] = value
        else:
            await logger.adebug(
                f"Filtering out context parameter '{key}' not found in context schema",
                context_key=key,
                available_keys=list(properties.keys()),
            )

    return filtered_context
