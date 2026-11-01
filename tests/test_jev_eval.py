"""GMA-446: pure helpers in the Jev eval scripts. No model load, no ~/.cache."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "model_eval"))
import jev_a2
import jev_score
from PIL import Image, ImageDraw


def test_tag_groups_and_regions_keep_negative_coords():
    raw = "![text+](-12,3,40,-5)\nhello\n![title](0,1,2,3)\nT"
    match = jev_score.TAG.search(raw)
    assert match.groups() == ("text", "-12", "3", "40", "-5")
    assert jev_score.regions(raw) == [
        ("text", (-12, 3, 40, -5), "hello"),
        ("title", (0, 1, 2, 3), "T"),
    ]
    assert jev_score.page_text(raw) == "hello\n\nT"


def test_drop_indexes_negative_coord_blocks_like_regions():
    raw = "pre\n![text](-10,0,10,10)\nkeep\n![text](0,-4,8,8)\ndrop"
    assert [bb for _, bb, _ in jev_score.regions(raw)] == [(-10, 0, 10, 10), (0, -4, 8, 8)]
    text, n = jev_a2._drop(raw, {0: 0.9, 1: 0.1}, 0.5)
    assert n == 1
    assert text.startswith("pre") and "keep" in text and "drop" not in text
    assert jev_a2._drop("plain", {0: 0.0}, 0.5) == ("plain", 0)


def test_drop_keeps_blocks_without_a_prob():
    raw = "![text](0,0,1,1)\nkeep\n![image](2,2,3,3)\npic"
    assert jev_a2._drop(raw, {0: 0.5}, 0.5) == (raw, 0)


def test_norm_bbox_clamps_and_orders_corners():
    assert jev_score.norm_bbox((-20, 1200, 40, -5)) == (0, 0, 40, 1000)
    assert jev_score.norm_bbox((900, 800, 100, 50)) == (100, 50, 900, 800)
    assert jev_score.norm_bbox((1, 2, 3, 4)) == (1, 2, 3, 4)


def test_crop_swapped_corners_is_the_normalized_region():
    img = Image.new("RGB", (1000, 1000), "blue")
    ImageDraw.Draw(img).rectangle((200, 100, 799, 899), fill="red")
    cropped = jev_score.crop(img, (800, 900, 200, 100), pad=0)
    assert cropped.size == (600, 800)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)
    assert cropped.getpixel((599, 799)) == (255, 0, 0)


def test_crop_degenerate_grows_around_center_not_full_page():
    img = Image.new("RGB", (100, 100), "white")
    img.putpixel((50, 40), (1, 2, 3))
    cropped = jev_score.crop(img, (500, 400, 500, 400), pad=0)
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == (1, 2, 3)
    assert cropped.size != img.size


def test_crop_clamped_point_stays_inside_the_image():
    img = Image.new("RGB", (100, 80), "white")
    img.putpixel((99, 0), (7, 8, 9))
    assert jev_score.norm_bbox((5000, -20, 9000, -1)) == (1000, 0, 1000, 0)
    cropped = jev_score.crop(img, (5000, -20, 9000, -1), pad=0)
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == (7, 8, 9)


def test_crop_thin_and_negative_boxes_are_not_the_full_page():
    img = Image.new("RGB", (1000, 1000), "white")
    assert jev_score.crop(img, (100, 0, 102, 1000), pad=0).size == (2, 1000)
    assert jev_score.crop(img, (-100, -100, 100, 200), pad=0).size == (100, 200)
    assert jev_score.crop(img, (0, 0, 0, 0)).size == (6, 6)
    assert jev_score.crop(img, (1000, 1000, 1000, 1000)).size == (6, 6)
    assert jev_score.crop(img, (100, 100, 200, 200), pad=6).size == (112, 112)


@pytest.mark.parametrize(("width", "expect"), [(1, 400), (400, 400), (1000, 1000), (1400, 1400), (2000, 1400)])
def test_render_width(width, expect):
    assert jev_score.render_width(Image.new("RGB", (width, 10))) == expect


def test_resume_jsonl_good_file():
    raw = b'{"key":"a/p0.json"}\n{"key":"b/p1.json"}\n'
    assert jev_score.resume_jsonl(raw) == ({"a/p0.json", "b/p1.json"}, len(raw))


def test_resume_jsonl_truncated_last_line(tmp_path):
    good = '{"key":"café"}\n'.encode()
    raw = good + b'{"key":"b"'
    done, end = jev_score.resume_jsonl(raw)
    assert done == {"café"} and end == len(good)
    assert jev_score.resume_jsonl(b'{"key":"only-partial"') == (set(), 0)
    path = tmp_path / "out.jsonl"
    path.write_bytes(raw)
    with path.open("rb+") as fh:
        fh.truncate(end)
    assert path.read_bytes() == good


def test_resume_jsonl_malformed_last_line_keeps_prior():
    good = b'{"key":"a"}\n'
    assert jev_score.resume_jsonl(good + b"{bad}\n") == ({"a"}, len(good))


def test_resume_jsonl_malformed_middle_line_raises():
    with pytest.raises(json.JSONDecodeError):
        jev_score.resume_jsonl(b'{bad\n{"key":"a"}\n{"key":"trail')
    with pytest.raises(KeyError):
        jev_score.resume_jsonl(b'{"key":"a"}\n{}\n{"key":"c"}\n')


def test_accepted_cer_uses_scored_pages_and_counts_unscored():
    cer = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 0.25}
    prob = {"a": 1.0, "b": 0.0, "c": None}
    assert jev_a2.accepted_cer(cer, prob, 0.5) == {
        "cer_all": 0.25, "cer_accepted": 0.0, "accepted": 1, "of": 2, "unscored": 2,
    }
    tied = jev_a2.accepted_cer({"a": 0.5}, {"a": 0.5}, 0.5)
    assert tied["accepted"] == 1 and tied["of"] == 1 and tied["unscored"] == 0
    empty = jev_a2.accepted_cer({"a": 0.2}, {"a": None}, 0.5)
    assert empty["cer_all"] is None and empty["cer_accepted"] is None
    assert empty["accepted"] == 0 and empty["of"] == 0 and empty["unscored"] == 1


def test_missing_score_count():
    keys = ["a/p0.json", "a/p1.json", "b/p0.json"]
    assert jev_a2.missing_score_count(keys, {"a/p0.json": {0: 0.5}}) == 2
    assert jev_a2.missing_score_count(keys, set(keys)) == 0
    assert jev_a2.missing_score_count([], {"a": 1}) == 0
