import json

from conftest import FakeLLM, response, text_block, tool_use_block

from docqa.agents import AgentOrchestrator, apply_validation
from docqa.guardrails import NOT_FOUND_MESSAGE, UNSAFE_MESSAGE
from docqa.rag import answer_with_rag


def plan(intent="factual_lookup", queries=("annual leave days",), direct=""):
    return response(
        text_block(
            json.dumps({"intent": intent, "sub_queries": list(queries), "approach": "Look it up.", "direct_response": direct})
        )
    )


def validation(grounded=True, revised="", unsafe=False):
    return response(
        text_block(
            json.dumps(
                {
                    "grounded": grounded,
                    "confidence": "high" if grounded else "low",
                    "unsupported_claims": [] if grounded else ["made-up claim"],
                    "contains_unsafe_content": unsafe,
                    "revised_answer": revised,
                    "notes": "",
                }
            )
        )
    )


def test_full_agent_workflow(store, settings):
    llm = FakeLLM(
        [
            plan(),
            response(
                tool_use_block("t1", "search_documents", {"query": "annual leave days"}), stop_reason="tool_use"
            ),
            response(text_block("Employees get 24 days of annual leave [S1][S42].")),
            validation(),
        ]
    )
    answer = AgentOrchestrator(store, llm, settings).run("How much annual leave do I get?")

    assert answer.text == "Employees get 24 days of annual leave [S1]."  # hallucinated [S42] removed
    assert [p.label for p in answer.passages] == ["S1"]
    assert "24 days" in answer.passages[0].text
    assert answer.validation["grounded"] is True

    # the search tool's passages were sent back to the model as a tool_result
    tool_results = llm.calls[2]["messages"][-1]["content"]
    assert tool_results[0]["tool_use_id"] == "t1"
    assert '<passage id="S1"' in tool_results[0]["content"]
    # the validator saw the retrieved passages
    assert "24 days" in llm.calls[3]["messages"][0]["content"]


def test_chit_chat_skips_retrieval(store, settings):
    llm = FakeLLM([plan(intent="chit_chat", queries=(), direct="Hello! Ask me about your documents.")])
    answer = AgentOrchestrator(store, llm, settings).run("hi there")
    assert answer.text == "Hello! Ask me about your documents."
    assert len(llm.calls) == 1


def test_no_evidence_returns_not_found_without_validation(store, settings):
    llm = FakeLLM(
        [
            plan(queries=("chocolate cake recipe",)),
            response(tool_use_block("t1", "search_documents", {"query": "chocolate cake frosting recipe"})),
            response(text_block("Mix flour and sugar...")),
        ]
    )
    settings = settings.__class__(**{**settings.__dict__, "min_relevance": 0.45})
    answer = AgentOrchestrator(store, llm, settings).run("How do I bake a cake?")
    assert answer.text == NOT_FOUND_MESSAGE
    assert len(llm.calls) == 3  # planner + 2 research turns, no validator call


def test_bad_tool_input_is_reported_to_the_model(store, settings):
    llm = FakeLLM(
        [
            plan(),
            response(tool_use_block("t1", "search_documents", {"query": "leave", "source": "missing.pdf"})),
            response(tool_use_block("t2", "search_documents", {"query": "annual leave"})),
            response(text_block("24 days [S1].")),
            validation(),
        ]
    )
    answer = AgentOrchestrator(store, llm, settings).run("Leave?")
    error_result = llm.calls[2]["messages"][-1]["content"][0]
    assert error_result["is_error"] is True and "missing.pdf" in error_result["content"]
    assert answer.text == "24 days [S1]."


def test_step_limit_forces_final_answer(store, settings):
    search = lambda i: response(tool_use_block(f"t{i}", "search_documents", {"query": f"leave {i}"}))
    llm = FakeLLM([plan(), search(1), search(2), search(3), response(text_block("24 days [S1].")), validation()])
    answer = AgentOrchestrator(store, llm, settings).run("Leave?")
    assert answer.text == "24 days [S1]."
    assert llm.calls[4]["tool_choice"] == {"type": "none"}


def test_apply_validation_rules():
    assert apply_validation("draft", {"grounded": True, "contains_unsafe_content": False}) == "draft"
    assert apply_validation("draft", {"grounded": False, "revised_answer": "fixed"}) == "fixed"
    assert "couldn't confirm" in apply_validation("draft", {"grounded": False, "revised_answer": ""})
    assert apply_validation("draft", {"grounded": True, "contains_unsafe_content": True}) == UNSAFE_MESSAGE


def test_simple_rag_pipeline(store, settings):
    llm = FakeLLM([response(text_block("Hotels are capped at 150 USD per night [S1]."))])
    answer = answer_with_rag("What is the hotel limit?", store, llm, settings, top_k=2)
    assert answer.text.endswith("[S1].")
    assert answer.passages[0].source == "travel.txt"
    assert "<passages>" in llm.calls[0]["messages"][0]["content"]


def test_simple_rag_skips_llm_when_nothing_relevant(store, settings):
    llm = FakeLLM([])
    settings = settings.__class__(**{**settings.__dict__, "min_relevance": 0.45})
    answer = answer_with_rag("chocolate cake frosting recipe", store, llm, settings)
    assert answer.text == NOT_FOUND_MESSAGE and llm.calls == []
