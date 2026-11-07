from reglens.ingest import clean_page, load_sources


def test_sources_config_is_valid():
    sources = load_sources()
    assert len(sources) >= 5
    assert len({s.id for s in sources}) == len(sources)
    assert all(s.url.startswith("https://rbidocs.rbi.org.in/") for s in sources)


def test_clean_page_drops_page_numbers_and_joins_hyphens():
    raw = "The regulated en-\ntity shall   verify\n\nPage 12 of 140\n  12  \nthe customer."
    out = clean_page(raw)
    assert "entity shall verify" in out
    assert "Page 12" not in out
    assert "\n12\n" not in out


def test_clean_page_empty():
    assert clean_page("") == ""
