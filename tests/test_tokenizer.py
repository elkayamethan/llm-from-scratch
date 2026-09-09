from data.tokenizer import GPT2Tokenizer, Tokenizer


def test_satisfies_protocol() -> None:
    assert isinstance(GPT2Tokenizer(), Tokenizer)


def test_vocab_and_eot() -> None:
    tok = GPT2Tokenizer()
    assert tok.vocab_size == 50257
    assert tok.eot_id == 50256


def test_roundtrip() -> None:
    tok = GPT2Tokenizer()
    text = "Hello world. Ünïcödé."
    assert tok.decode(tok.encode(text)) == text


def test_literal_eot_is_not_special() -> None:
    tok = GPT2Tokenizer()
    text = "before <|endoftext|> after"
    ids = tok.encode(text)
    assert tok.eot_id not in ids
    assert tok.decode(ids) == text
