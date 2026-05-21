import pytest
from aegra_api.utils.run_utils import sanitize_for_db

def test_sanitize_string_with_nul():
    assert sanitize_for_db("hello\u0000world") == "helloworld"

def test_sanitize_string_with_surrogates():
    # Unpaired high surrogate
    bad_str = "hello\ud83dworld"
    sanitized = sanitize_for_db(bad_str)
    assert sanitized == "helloworld"

def test_sanitize_nested_dict():
    data = {
        "input": {"text": "value\u00001"},
        "config": ["item1", "item\ud83d2"]
    }
    expected = {
        "input": {"text": "value1"},
        "config": ["item1", "item2"]
    }
    assert sanitize_for_db(data) == expected

def test_sanitize_bytes():
    # Null byte in bytes
    bad_bytes = b"hello\x00world"
    assert sanitize_for_db(bad_bytes) == "helloworld"
    
    # Invalid utf-8 bytes
    invalid_bytes = b"hello\xffworld"
    assert sanitize_for_db(invalid_bytes) == "helloworld"

def test_sanitize_none_and_other_types():
    assert sanitize_for_db(None) is None
    assert sanitize_for_db(123) == 123
