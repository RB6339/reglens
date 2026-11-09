from reglens.answer import REFUSAL, Answerer, build_prompt
from reglens.retrieve import BM25


def _chunk(para, text, page):
    return {"id": f"kyc:{para}:0", "source": "kyc", "title": "KYC Directions", "chapter": "",
            "para": para, "page_start": page, "page_end": page, "text": text}


CHUNKS = [
    _chunk("42", "Periodic updation of KYC shall be carried out at least once every two years for high risk "
                 "customers, once every eight years for medium risk and once every ten years for low risk.", 41),
    _chunk("26", "The bank may undertake V-CIP (video based customer identification process) for onboarding.", 30),
]


def _answerer(llm):
    # the 2-chunk test corpus gives small BM25 scores, so use a lower refusal threshold
    return Answerer(BM25(CHUNKS), llm, min_score=0.5)


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, system, user):
        self.prompts.append(user)
        return self.reply


def test_answer_keeps_valid_citations_and_maps_them():
    llm = FakeLLM("High risk customers need updation every two years [1]. See also [7].")
    ans = _answerer(llm).ask("how often is KYC periodic updation for high risk customers")
    assert not ans.refused
    assert "[7]" not in ans.text  # invented block number dropped
    assert ans.citations == [{"n": 1, "citation": "KYC Directions, para 42, p. 41", "chunk_id": "kyc:42:0"}]
    assert "[1] (KYC Directions, para 42, p. 41)" in llm.prompts[0]


def test_refuses_without_calling_llm_when_nothing_relevant():
    llm = FakeLLM("should not be called")
    ans = _answerer(llm).ask("what is the repo rate for cryptocurrency loans")
    assert ans.refused and ans.text == REFUSAL
    assert llm.prompts == []


def test_refuses_when_model_says_not_covered_or_cites_nothing():
    q = "periodic updation of KYC for high risk customers"
    assert _answerer(FakeLLM("NOT_COVERED")).ask(q).refused
    assert _answerer(FakeLLM("Every two years.")).ask(q).refused


def test_prompt_numbers_blocks_in_order():
    hits = BM25(CHUNKS).search("KYC V-CIP updation", k=2)
    prompt = build_prompt("q", hits)
    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.endswith("Question: q\nAnswer:")
