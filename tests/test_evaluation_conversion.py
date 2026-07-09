from __future__ import annotations

import json
from pathlib import Path


def test_stm_to_txt_and_txt_to_stm_roundtrip_with_sidecar(tmp_path: Path) -> None:
    from sure_eval.evaluation.conversion.sa_asr__cpwer.stm_to_txt import convert_stm_to_txt
    from sure_eval.evaluation.conversion.sa_asr__cpwer.txt_to_stm import convert_txt_to_stm

    stm_path = tmp_path / "input.stm"
    txt_path = tmp_path / "text.txt"
    sidecar_path = tmp_path / "sidecar.json"
    restored_path = tmp_path / "restored.stm"
    stm_path.write_text(
        "\n".join(
            [
                "rec1 1 spk1 0.00 3.00 Hello, world!",
                "rec1 1 spk2 3.00 6.00 Good morning.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trace = convert_stm_to_txt(
        input_stm=str(stm_path),
        output_txt=str(txt_path),
        sidecar_json=str(sidecar_path),
        conversion_id="sa_asr__cpwer",
    )

    assert trace["id"] == "sa_asr__cpwer"
    assert trace["source_format"] == "stm"
    assert trace["target_format"] == "key_text"
    assert trace["affects_metric"] is True
    txt_rows = txt_path.read_text(encoding="utf-8").splitlines()
    assert txt_rows == [
        "rec1|1|spk1|0.00|3.00|0\tHello, world!",
        "rec1|1|spk2|3.00|6.00|1\tGood morning.",
    ]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["rows"][0] == {
        "key": "rec1|1|spk1|0.00|3.00|0",
        "session_id": "rec1",
        "channel": "1",
        "speaker_id": "spk1",
        "start": "0.00",
        "end": "3.00",
        "row_index": 0,
    }

    txt_path.write_text(
        "\n".join(
            [
                "rec1|1|spk1|0.00|3.00|0\tHELLO WORLD",
                "rec1|1|spk2|3.00|6.00|1\tGOOD MORNING",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    restore_trace = convert_txt_to_stm(
        input_txt=str(txt_path),
        sidecar_json=str(sidecar_path),
        output_stm=str(restored_path),
        conversion_id="sa_asr__cpwer",
    )

    assert restore_trace["source_format"] == "key_text"
    assert restore_trace["target_format"] == "stm"
    assert restored_path.read_text(encoding="utf-8").splitlines() == [
        "rec1 1 spk1 0.00 3.00 HELLO WORLD",
        "rec1 1 spk2 3.00 6.00 GOOD MORNING",
    ]


def test_sa_asr_cpwer_conversion_profile_is_task_metric_scoped() -> None:
    profile_dir = Path("src/sure_eval/evaluation/conversion/sa_asr__cpwer")

    assert profile_dir.exists()
    assert (profile_dir / "manifest.yaml").read_text(encoding="utf-8").startswith("id: sa_asr__cpwer")
    assert (profile_dir / "stm_to_txt.py").exists()
    assert (profile_dir / "txt_to_stm.py").exists()
    assert not Path("src/sure_eval/evaluation/conversion/default__sa_asr__cpwer").exists()


def test_gstar_norm_operates_on_key_text_files_only(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.gstar_norm import normalize_gstar_sa_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\tHello, world!\n", encoding="utf-8")
    hyp_file.write_text("utt1\thello world\n", encoding="utf-8")

    normalized_files, trace = normalize_gstar_sa_asr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        language="en",
    )

    assert trace.node_id == "normalization/gstar_norm"
    assert trace.details["input_schema"] == "key_text_files"
    assert trace.details["output_schema"] == "key_text_files"
    assert Path(normalized_files.ref_file).read_text(encoding="utf-8") == "utt1\tHELLO WORLD\n"
