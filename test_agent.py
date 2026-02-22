from Agents.graph import invoke_agent
import uuid

FIXED_USER = "11111111-1111-1111-1111-111111111111"


def run_test(name, thread_id, queries, expected_checks):
    print(f"\n===== TEST: {name} =====")

    state = None
    for q in queries:
        state = invoke_agent(
            user_query=q,
            thread_id=thread_id,
            user_id=FIXED_USER,
        )

    results = []

    for check_name, check_fn in expected_checks.items():
        try:
            passed = check_fn(state)
            results.append((check_name, passed))
        except Exception as e:
            results.append((check_name, False))
            print(f"ERROR in {check_name}: {e}")

    for check_name, passed in results:
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    return all(passed for _, passed in results)


def test_confirmation_yes():
    return run_test(
        name="Multi-intent Confirmation YES",
        thread_id="test_confirm_yes",
        queries=[
            "Install VS 2024 and my VPN is not working",
            "yes",
        ],
        expected_checks={
            "Confirmed True": lambda s: s.get("confirmed") is True,
            "Two tasks completed": lambda s: len(s.get("tasks", [])) == 2
            and all(t["status"] == "COMPLETED" for t in s["tasks"]),
            "Status completed": lambda s: s.get("status") == "COMPLETED",
        },
    )


def test_confirmation_no():
    return run_test(
        name="Multi-intent Confirmation NO",
        thread_id="test_confirm_no",
        queries=[
            "Install VS 2024 and my VPN is not working",
            "no",
        ],
        expected_checks={
            "Tasks cleared": lambda s: len(s.get("tasks", [])) == 0,
            "Status completed": lambda s: s.get("status") == "COMPLETED",
        },
    )


def test_single_intent():
    return run_test(
        name="Single Intent Ticket",
        thread_id="test_single_intent",
        queries=["VPN is not working"],
        expected_checks={
            "Ticket created": lambda s: s.get("ticket_id") is not None,
            "Status completed": lambda s: s.get("status") == "COMPLETED",
        },
    )


def test_rag_valid():
    return run_test(
        name="RAG Valid Policy",
        thread_id="test_rag_valid",
        queries=["What is the payroll processing date?"],
        expected_checks={
            "RAG found": lambda s: s.get("rag_found") is True,
            "RAG context not empty": lambda s: bool(s.get("rag_context")),
        },
    )


def test_rag_invalid():
    return run_test(
        name="RAG Invalid Policy",
        thread_id="test_rag_invalid",
        queries=["What is the Mars relocation policy?"],
        expected_checks={
            "RAG not found": lambda s: s.get("rag_found") is False,
        },
    )


if __name__ == "__main__":
    tests = [
        test_confirmation_yes,
        test_confirmation_no,
        test_single_intent,
        test_rag_valid,
        test_rag_invalid,
    ]

    passed = 0

    for test in tests:
        if test():
            passed += 1

    print(f"\n===== RESULT: {passed}/{len(tests)} TESTS PASSED =====")

