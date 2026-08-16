import pytest


@pytest.fixture(scope="session")
def clean_text(app_module):
    return app_module.clean_text


def test_strips_inr_amount(clean_text):
    assert "inr" not in clean_text("POS PURCH INR 1418")


def test_strips_transaction_id(clean_text):
    assert "txn" not in clean_text("Wallet pymt TXN69cd51f6")


def test_strips_bare_digits(clean_text):
    assert not any(c.isdigit() for c in clean_text("DTH RECHRGE 1235"))


def test_lowercases(clean_text):
    assert clean_text("SHELL PETROL") == "shell petrol"


def test_trims_surrounding_whitespace(clean_text):
    assert clean_text("   Uber ride   ") == "uber ride"


def test_full_row_keeps_only_the_merchant_words(clean_text):
    assert clean_text("POS PURCH INR 1418 TXN9c72946c") == "pos purch"


def test_empty_string(clean_text):
    assert clean_text("") == ""


def test_digits_only(clean_text):
    assert clean_text("INR 900 TXNabc123") == ""


def test_is_idempotent(clean_text):
    once = clean_text("Debit Card Purr INR 1208 TXN9a34baa4")
    assert clean_text(once) == once


def test_preserves_merchant_name(clean_text):
    assert "netflix" in clean_text("NETFLIX SUBSCRIPTION INR 499")
