from main import main


def test_main():
    load_model = main()
    assert load_model is not None


if __name__ == "__main__":
    test_main()
