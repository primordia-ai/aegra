import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aegra_api.api.runs import update_run_status
from aegra_api.core.orm import Run as RunORM
from sqlalchemy import update

@pytest.mark.asyncio
async def test_update_run_status_sanitization_integration():
    """
    Integration test for update_run_status to ensure it sanitizes output and error
    before committing to the database.
    """
    # 1. Setup mock session and update
    mock_session = AsyncMock()
    mock_update_stmt = MagicMock()
    
    # problematic data
    run_id = "test-run-123"
    status = "success"
    bad_output = {
        "result": "Clean",
        "problem": "NUL\u0000Byte and Surrogate\ud83d"
    }
    bad_error = "Error with NUL\u0000"
    
    # Mock the update call chain: update(RunORM).where(...).values(...)
    with patch("aegra_api.api.runs.update", return_value=mock_update_stmt), \
         patch("aegra_api.api.runs.validate_run_status", side_effect=lambda x: x):
        
        mock_update_stmt.where.return_value = mock_update_stmt
        mock_update_stmt.values.return_value = mock_update_stmt
        
        # 3. Call update_run_status
        await update_run_status(
            run_id=run_id,
            status=status,
            output=bad_output,
            error=bad_error,
            session=mock_session
        )
    
    # 4. Verify values() was called with sanitized data
    values_call = mock_update_stmt.values.call_args
    assert values_call is not None
    
    values_passed = values_call.kwargs
    
    assert "output" in values_passed
    assert "error_message" in values_passed
    
    sanitized_output = values_passed["output"]
    sanitized_error = values_passed["error_message"]
    
    # Verify NUL bytes and surrogates are gone
    # output is a serialized JSON string or dict
    output_str = str(sanitized_output)
    assert "\u0000" not in output_str
    assert "\ud83d" not in output_str
    assert "\u0000" not in sanitized_error
    
    # Verify session.execute was called
    mock_session.execute.assert_called_once_with(mock_update_stmt)
    # Verify session.commit was called
    mock_session.commit.assert_called_once()
