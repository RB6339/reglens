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


def test_table_of_contents_lines_are_skipped():
    pages = _pages(
        "Table of Contents\n"
        "Chapter I – Preliminary ........................................ 4\n"
        "A. Short Title and Commencement. ................................ 4\n"
        "Chapter VI – Customer Due Diligence\n"
        "16. The bank shall undertake identification of customers before opening an account.",
    )
    chunks = chunk_document(pages)
    assert all("....." not in c.text for c in chunks)
    assert chunks[-1].chapter == "Chapter VI - Customer Due Diligence"


def test_chunk_ids_are_unique():
    pages = _pages(
        "Some preamble text that is long enough to become a chunk on its own.\n"
        "Chapter I – Preliminary\nMore preamble text under chapter one, also long enough to keep.\n"
        "Chapter II – General\nAnd a third preamble block under chapter two that is long enough.",
    )
    ids = [c.id for c in chunk_document(pages)]
    assert len(ids) == len(set(ids))


def test_annex_paragraphs_get_their_own_labels():
    pages = _pages(
        "Chapter XI – Repeal and Other Provisions\n"
        "2. The provisions of these Directions shall be in addition to any other law in force.\n"
        "Annex – I\n"
        "2. The procedure for freezing of funds of designated individuals is outlined below.",
    )
    chunks = chunk_document(pages)
    main, annex = chunks[0], chunks[-1]
    assert (main.para, main.annex) == ("2", "")
    assert (annex.para, annex.annex, annex.chapter) == ("2", "I", "Annex I")
    assert main.id != annex.id and annex.id.startswith("kyc:annex-i-2")


def test_text_after_chapter_heading_is_not_the_previous_para():
    pages = _pages(
        "7. The Board shall review the KYC policy at regular intervals of time.\n"
        "Chapter III – Customer Acceptance Policy\n"
        "This chapter sets out the minimum requirements for accepting new customers.",
    )
    assert chunk_document(pages)[-1].para == "intro"
