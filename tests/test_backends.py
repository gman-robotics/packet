import pytest

from packet.backends import (
    FallbackExtractor,
    LightOnExtractor,
    PaddleExtractor,
    default_ocr_extractor,
)
from packet.types import PageClass, PageRef

REF = PageRef(packet_id="p", index=0, page_class=PageClass.SCAN)


class _Ok:
    def supports(self, pc):
        return True

    def extract(self, ref, text):
        return ["ok"]


class _Boom(_Ok):
    def extract(self, ref, text):
        raise RuntimeError("down")


def test_default_is_lighton_grounding_mlx_with_paddle_fallback():
    ext = default_ocr_extractor()
    assert isinstance(ext.primary, LightOnExtractor)
    assert ext.primary.mode == "grounding" and ext.primary.backend == "mlx"
    assert ext.primary.model.endswith("LightOnOCR-3-1B")
    assert isinstance(ext.fallback, PaddleExtractor)
    assert ext.supports(PageClass.SCAN)


def test_primary_used_when_it_works():
    ext = FallbackExtractor(_Ok(), _Boom())
    assert ext.extract(REF, "") == ["ok"]
    assert isinstance(ext.last_used, _Ok)


def test_falls_back_when_primary_fails():
    ext = FallbackExtractor(_Boom(), _Ok())
    assert ext.extract(REF, "") == ["ok"]
    assert ext.last_used is ext.fallback


def test_fallback_error_propagates():
    with pytest.raises(NotImplementedError):
        default_ocr_extractor().extract(REF, "")
