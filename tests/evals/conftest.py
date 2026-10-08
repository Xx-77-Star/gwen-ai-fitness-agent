"""Pytest reporting for local Agent Evaluation metrics."""


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    from tests.evals.harness import latest_tool_selection_report
    from tests.evals.response_harness import latest_response_quality_report

    tool_report = latest_tool_selection_report()
    if tool_report is not None:
        terminalreporter.section("Tool Selection Evaluation")
        for line in tool_report.summary_lines():
            terminalreporter.write_line(line)
        for prediction in tool_report.failures:
            terminalreporter.write_line(
                f"FAIL {prediction.case.id}: expected={prediction.case.expected_tool} "
                f"actual={prediction.selected_tool} stop_reason={prediction.stop_reason} "
                f"error={prediction.error_code}"
            )

    response_report = latest_response_quality_report()
    if response_report is not None:
        terminalreporter.section("Response Quality Evaluation")
        for line in response_report.summary_lines():
            terminalreporter.write_line(line)
        for prediction in response_report.failures:
            terminalreporter.write_line(
                f"FAIL {prediction.case.id}: score={prediction.score:.2%} "
                f"dimensions={prediction.dimension_scores}"
            )