"""Task-level privilege for the loss-1 text teacher (train only).

"""

from __future__ import annotations

from realmg.eval.q_para_prompts import assert_task_privilege_has_no_closed_set_label


R_CNT_TEACHER_TASK_PRIVILEGE = (
    "Answer concisely and accurately. Focus on the question itself. "
    "Avoid redundancy, boilerplate, and lengthy tutorials."
)


def attach_r_cnt_teacher_privilege(question_text: str) -> str:
    """Prepend the concise-answer privilege to the teacher's text(s) question."""

    assert_task_privilege_has_no_closed_set_label(R_CNT_TEACHER_TASK_PRIVILEGE)
    privilege = R_CNT_TEACHER_TASK_PRIVILEGE.strip()
    question = question_text.strip()
    if not question:
        raise ValueError("R-cnt teacher question text is empty.")
    if question.startswith(privilege):
        return question
    return f"{privilege}\n\n{question}"
