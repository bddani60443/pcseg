import pytest

from pcseg.cli import build_parser


def test_run_subcommand_parses_minimal_args():
    parser = build_parser()
    args = parser.parse_args([
        "run", "--input", "scan.e57", "--prompt", "door", "--out", "result.las",
    ])
    assert args.command == "run"
    assert args.input == "scan.e57"
    assert args.outputs == ["result.las"]
    assert args.dino_channel == "auto"
    assert args.filter == "none"


def test_run_subcommand_device_flag_defaults_to_auto_and_accepts_cpu():
    parser = build_parser()
    args = parser.parse_args(["run", "-i", "scan.e57", "-p", "door", "-o", "result.las"])
    assert args.device == "auto"

    args = parser.parse_args(["run", "-i", "scan.e57", "-p", "door", "-o", "result.las", "--device", "cpu"])
    assert args.device == "cpu"


def test_run_subcommand_supports_repeated_out_and_pitch_list():
    parser = build_parser()
    args = parser.parse_args([
        "run", "-i", "scan.e57", "-p", "door, window",
        "-o", "result.las", "-o", "preview.png",
        "--pitch", "0", "30", "-30",
    ])
    assert args.outputs == ["result.las", "preview.png"]
    assert args.pitch == [0.0, 30.0, -30.0]


def test_run_requires_input_prompt_and_out():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_no_command_is_required_error():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
