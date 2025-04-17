# Outstanding Issues

This file tracks known issues, bugs, and areas for improvement that need to be addressed.

## Linting Issues (from 2025-04-17 check)

---

**Issue 1: Unused Variable `batch_throughput`**

*   **File:** `src/advanced_calculator/modules/throughput.py`
*   **Line:** 194
*   **Error:** F841 Local variable `batch_throughput` is assigned to but never used
*   **Suggestion:** Remove assignment to unused variable `batch_throughput`

---

**Issue 2: Unused Variables in Web App Route**

*   **File:** `src/advanced_calculator/web/web_app.py`
*   **Lines:** 282-285
*   **Errors:** F841 Local variables `hidden_dim`, `ff_dim`, `num_layers`, `vocab_size` are assigned to but never used
*   **Suggestion:** Remove assignments to these unused variables if they are truly not needed, or utilize them if they were intended for calculation/logging.

---

**Issue 3: Module Import Not at Top**

*   **File:** `src/gui/main.py`
*   **Line:** 13
*   **Error:** E402 Module level import `from src.advanced_calculator.main import AdvancedCalculator` not at top of file
*   **Suggestion:** Move this import statement to the top of the file with other imports.

---

**Issue 4: Module Imports Not at Top**

*   **File:** `tests/benchmark_llm_api.py`
*   **Lines:** 11, 12
*   **Errors:**
    *   E402 Module level import `import tiktoken` not at top of file
    *   E402 Module level import `from src.advanced_calculator.main import AdvancedCalculator` not at top of file
*   **Suggestion:** Move these import statements to the top of the file.

---

**Issue 5: Multiple Statements on One Line (GUI TFLOPS Logic)**

*   **File:** `src/gui/main.py`
*   **Lines:** 163, 164, 165, 166, 167
*   **Error:** E701 Multiple statements on one line (colon) within the `if/elif` block determining `tflops_key`.
*   **Suggestion:** Refactor each `if/elif` line to have the assignment on a separate, indented line.

---

**Issue 6: Multiple Statements on One Line (GUI Update Logic)**

*   **File:** `src/gui/main.py`
*   **Lines:** 230, 231, 232, 233, 234
*   **Error:** E701 Multiple statements on one line (colon) within the `if/elif` block determining `tflops_key` in the update function.
*   **Suggestion:** Refactor each `if/elif` line to have the assignment on a separate, indented line.

---

**Issue 7: Bare Except Block**

*   **File:** `tests/benchmark_llm_api.py`
*   **Line:** 130
*   **Error:** E722 Do not use bare `except`
*   **Suggestion:** Specify the exact exception type(s) that are expected (e.g., `except Exception:`, or ideally a more specific exception like `KeyError` or `ValueError` if applicable).

--- 