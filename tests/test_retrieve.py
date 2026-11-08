from reglens.retrieve import BM25, Hit, tokenize


def _chunk(para, text, page=1, source="kyc", chapter=""):
    return {
        "id": f"{source}:{para}:0", "source": source, "title": "KYC Directions", "chapter": chapter,
        "para": para, "page_start": page, "page_end": page, "text": text,
    }


CHUNKS = [
    _chunk("38", "Periodic updation of KYC shall be carried out at least once every two years for high "
                 "risk customers and once every ten years for low risk customers.", page=30),
    _chunk("18", "Video based Customer Identification Process (V-CIP) may be undertaken by the RE.", page=12),
    _chunk("3", "Officially Valid Document means the passport, the driving licence and proof of possession "
                "of Aadhaar number.", page=4, chapter="Chapter I - Preliminary"),
    _chunk("5", "Interest rates on deposits shall be decided by the bank board.", source="deposit_rates"),
]


def test_tokenize_keeps_hyphenated_terms_and_drops_stopwords():
    assert tokenize("The V-CIP of a customer is re-KYC") == ["v-cip", "customer", "re-kyc"]


def test_relevant_paragraph_ranks_first():
    hits = BM25(CHUNKS).search("how often is KYC updated for high risk customers")
    assert hits[0].chunk["para"] == "38"


def test_source_filter_and_no_match():
    index = BM25(CHUNKS)
    assert all(h.chunk["source"] == "deposit_rates" for h in index.search("interest rates", source="deposit_rates"))
    assert index.search("cryptocurrency") == []
    assert index.search("the of and") == []


def test_citation_format():
    hit = Hit(1.0, _chunk("38", "text", page=30))
    assert hit.citation == "KYC Directions, para 38, p. 30"
    span = Hit(1.0, {**_chunk("preamble", "text"), "page_start": 1, "page_end": 2})
    assert span.citation == "KYC Directions, preamble, p. 1-2"
