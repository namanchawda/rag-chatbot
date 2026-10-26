"""The app.* namespace must be visible under plain uvicorn: one StreamHandler
sits on the "app" logger with propagate=False, guarded against duplicates."""

import io
import logging

from app.main import configure_logging


def test_app_info_record_reaches_handler():
    # Idempotent: a second call (e.g. uvicorn --reload re-imports) must not
    # stack another handler.
    configure_logging()
    configure_logging()

    app_logger = logging.getLogger("app")
    assert app_logger.propagate is False
    handlers = [h for h in app_logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(handlers) == 1
    handler = handlers[0]

    capture = io.StringIO()
    previous_stream = handler.stream
    handler.stream = capture
    try:
        logging.getLogger("app.rag.retrieve").info("rerank score probe")
    finally:
        handler.stream = previous_stream

    output = capture.getvalue()
    assert "rerank score probe" in output
    assert "INFO" in output
    assert "app.rag.retrieve" in output
    # Simple format: time, level, logger name, then the message.
    line = output.strip()
    assert line.endswith("app.rag.retrieve: rerank score probe")
    fields = line.split()
    assert fields[2] == "INFO"
    assert fields[3] == "app.rag.retrieve:"
