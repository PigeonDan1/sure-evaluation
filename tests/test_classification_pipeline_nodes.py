from __future__ import annotations

import json
from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_classify_node_uses_dataset_label_spec_with_aliases_and_numeric_ids(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.classify import load_label_spec, score_classification_files

    spec_file = tmp_path / "ser_labels.yaml"
    spec_file.write_text(
        """
id: ser_default
task: SER
labels:
  - id: neu
    aliases: [neutral]
    numeric_ids: [0]
  - id: hap
    aliases: [happy, happiness]
    numeric_ids: [1]
  - id: ang
    aliases: [angry, anger]
    numeric_ids: [2]
  - id: sad
    aliases: [sadness]
    numeric_ids: [3]
unknown_policy: invalid
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hap"), ("utt2", "woman"), ("utt3", "ang")])
    _write_key_text(hyp_file, [("utt1", "happy"), ("utt2", "female"), ("utt3", "2")])

    _, result = score_classification_files(
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        label_spec=load_label_spec(spec_file),
    )

    assert result.node_id == "scoring/classify"
    assert result.details["result"]["accuracy"] == 2 / 3
    assert result.details["result"]["correct"] == 2
    assert result.details["result"]["total"] == 3
    assert result.details["result"]["per_sample"][0]["prediction"] == "hap"
    assert result.details["result"]["per_sample"][1]["prediction_valid"] is False


def test_prompt_norm_supports_variable_choice_ids_and_choice_counts(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.prompt_norm import normalize_prompt_choice_files

    prompt_file = tmp_path / "prompt.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "key": "q1",
                        "choices": [
                            {"id": "yes", "text": "是", "aliases": ["正确"]},
                            {"id": "no", "text": "否", "aliases": ["错误"]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "key": "q2",
                        "choices": {
                            "1": "北京",
                            "2": "上海",
                            "3": "广州",
                            "4": "深圳",
                            "5": "杭州",
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("q1", "yes"), ("q2", "3")])
    _write_key_text(hyp_file, [("q1", "答案是正确"), ("q2", "我认为是广州")])

    normalized, result = normalize_prompt_choice_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        prompt_jsonl=str(prompt_file),
    )

    assert result.node_id == "normalization/prompt_norm"
    assert Path(normalized.ref_file).read_text(encoding="utf-8").splitlines() == ["q1\tyes", "q2\t3"]
    assert Path(normalized.hyp_file).read_text(encoding="utf-8").splitlines() == ["q1\tyes", "q2\t3"]
    assert result.details["num_choices_by_key"] == {"q1": 2, "q2": 5}


def test_prompt_norm_legacy_prompt_text_matches_sure_evaluator_processing(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.prompt_norm import normalize_prompt_choice_files
    from sure_eval.evaluation.sure_evaluator import _process_slu_prediction_file

    prompt_file = tmp_path / "prompt.jsonl"
    prompt_file.write_text(
        json.dumps(
            {
                "key": "q1",
                "prompt": "请作答\nA. 北京\nB. 上海\nC. 广州\nD. 深圳",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("q1", "C")])
    _write_key_text(hyp_file, [("q1", "答案是 C")])

    legacy_ref = tmp_path / "legacy_ref.txt"
    legacy_hyp = tmp_path / "legacy_hyp.txt"
    _process_slu_prediction_file(str(prompt_file), str(ref_file), str(legacy_ref))
    _process_slu_prediction_file(str(prompt_file), str(hyp_file), str(legacy_hyp))

    normalized, _ = normalize_prompt_choice_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        prompt_jsonl=str(prompt_file),
        output_mode="choice_text",
    )

    assert Path(normalized.ref_file).read_text(encoding="utf-8") == legacy_ref.read_text(encoding="utf-8")
    assert Path(normalized.hyp_file).read_text(encoding="utf-8") == legacy_hyp.read_text(encoding="utf-8")


def test_classification_task_routes_ser_and_gr_through_classify_node(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.classification.pipeline import evaluate_classification_files
    from sure_eval.evaluation.tasks.classification.metrics import AccuracyMetric

    ref_file = tmp_path / "ref_ser.txt"
    hyp_file = tmp_path / "hyp_ser.txt"
    _write_key_text(ref_file, [("utt1", "hap"), ("utt2", "ang")])
    _write_key_text(hyp_file, [("utt1", "happy"), ("utt2", "2")])

    report = evaluate_classification_files(str(ref_file), str(hyp_file), task="SER")
    legacy = AccuracyMetric().calculate_batch(["happy", "2"], ["hap", "ang"], task="SER")

    assert report.task == "SER"
    assert report.metric == "accuracy"
    assert report.score == legacy.score
    assert report.pipeline_trace[0].node_id == "scoring/classify"
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref", "label_spec"]

    ref_gr_file = tmp_path / "ref_gr.txt"
    hyp_gr_file = tmp_path / "hyp_gr.txt"
    _write_key_text(ref_gr_file, [("utt1", "man"), ("utt2", "woman")])
    _write_key_text(hyp_gr_file, [("utt1", "0"), ("utt2", "female")])

    gr_report = evaluate_classification_files(str(ref_gr_file), str(hyp_gr_file), task="GR")
    assert gr_report.task == "GR"
    assert gr_report.score == 1.0


def test_slu_task_pipeline_matches_sure_evaluator(tmp_path: Path) -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.slu.pipeline import evaluate_slu_files

    prompt_file = tmp_path / "prompt_slu.jsonl"
    prompt_file.write_text(
        json.dumps(
            {
                "key": "q1",
                "prompt": "请作答\nA. 北京\nB. 上海\nC. 广州\nD. 深圳",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ref_file = tmp_path / "ref_slu.txt"
    hyp_file = tmp_path / "hyp_slu.txt"
    _write_key_text(ref_file, [("q1", "C")])
    _write_key_text(hyp_file, [("q1", "答案是 C")])

    legacy = SUREEvaluator(language="zh").evaluate(
        "slu",
        str(ref_file),
        str(hyp_file),
        prompt_jsonl=str(prompt_file),
    )
    report = evaluate_slu_files(str(ref_file), str(hyp_file), prompt_jsonl=str(prompt_file))

    assert report.task == "SLU"
    assert report.score == legacy
    assert report.pipeline_trace[0].node_id == "normalization/prompt_norm"
    assert report.pipeline_trace[1].node_id == "scoring/classify"
    assert not Path(report.details["normalized_files"]["ref"]).exists()
    assert not Path(report.details["normalized_files"]["hyp"]).exists()
