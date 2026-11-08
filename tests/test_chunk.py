from reglens.chunk import chunk_document


def _pages(*texts):
    return [{"source": "kyc", "title": "KYC Directions", "page": i, "text": t} for i, t in enumerate(texts, 1)]


def test_splits_on_numbered_paragraphs_and_tracks_chapter():
    pages = _pages(
        "Chapter VI – Customer Due Diligence\n"
        "18. The Customer Acceptance Policy shall not result in denial of banking facility.\n"
        "19. Where the bank forms a suspicion of money laundering it shall file an STR.",
    )
    chunks = chunk_document(pages)
    assert [c.para for c in chunks] == ["18", "19"]
    assert all(c.chapter == "Chapter VI - Customer Due Diligence" for c in chunks)


def test_paragraph_spanning_pages_keeps_page_range():
    pages = _pages(
        "42. Updation / Periodic Updation of KYC\n(1) The bank shall adopt a risk-based approach",
        "for periodic updation of KYC at least once in every two years for high-risk customers.",
    )
    (chunk,) = chunk_document(pages)
    assert chunk.para == "42"
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_long_paragraph_is_split_at_sub_items():
    body = "\n".join(f"({i}) " + "the bank shall verify the customer identity. " * 6 for i in range(1, 9))
    chunks = chunk_document(_pages("42. Periodic updation\n" + body), max_chars=600)
    assert len(chunks) > 1
    assert all(c.para == "42" for c in chunks)
    assert all(len(c.text) < 1200 for c in chunks)
