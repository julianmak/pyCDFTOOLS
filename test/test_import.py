import pytest
from pyCDFTOOLS import hello


def test_hello(capfd):
    # test the import of function hello
    hello()
    out, err = capfd.readouterr()
    assert out.replace('\n', '') == "hello"
