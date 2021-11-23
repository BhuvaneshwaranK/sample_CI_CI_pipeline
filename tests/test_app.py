from app import index


def test_index():
    assert index() == "Hello, world!"


def test_test():
    assert index() == "coming in test page"