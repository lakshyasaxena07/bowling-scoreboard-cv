"""Unit tests for OCR Normalization and Symbol Disambiguation."""

import pytest
from src.ocr_engine import ScoreboardOCREngine


def test_roll_symbol_normalization():
    ocr = ScoreboardOCREngine()

    assert ocr.normalize_roll_symbol("x") == "X"
    assert ocr.normalize_roll_symbol("X") == "X"
    assert ocr.normalize_roll_symbol("/") == "/"
    assert ocr.normalize_roll_symbol("-") == "-"
    assert ocr.normalize_roll_symbol("F") == "-"
    assert ocr.normalize_roll_symbol("O") == "0"
    assert ocr.normalize_roll_symbol("I") == "1"
    assert ocr.normalize_roll_symbol("S") == "5"
    assert ocr.normalize_roll_symbol("9") == "9"
    assert ocr.normalize_roll_symbol("") == ""
