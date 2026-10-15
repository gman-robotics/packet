"""GMA-218 (b): a weak topic/language-shift signal between pages."""

from packet.boundary import BoundaryDetector
from packet.topic import content_words, detect_language, topic_similarity
from packet.types import Block, BlockKind, PageRef, SourceKind

LEASE = ("The tenant shall pay rent monthly to the landlord. The premises include the "
         "apartment, parking space and storage unit. The landlord maintains the building, "
         "repairs plumbing and heating, and the tenant keeps the apartment clean. Rent "
         "increases require written notice. The lease term runs twelve months and renews "
         "unless the tenant gives notice. Security deposit returns after inspection of "
         "the premises by the landlord. Pets require landlord consent.")
LEASE2 = ("The tenant may not sublet the apartment without landlord consent. Rent is due on "
          "the first day; late rent incurs a fee. The landlord may enter the premises with "
          "notice for repairs. Utilities for heating and water are paid by the tenant. "
          "Parking space assignment follows the building rules. Storage unit access is "
          "limited. At the end of the lease term the tenant returns all keys to the landlord.")
REACTOR = ("Neutron flux measurements in the reactor core show thermal gradients across "
           "fuel assemblies. Control rods absorb neutrons and moderate the fission chain "
           "reaction. Coolant circulation removes heat through the primary loop into steam "
           "generators. Isotope inventories were computed with Monte Carlo transport codes "
           "and benchmarked against calorimetry experiments of irradiated uranium samples "
           "with burnup estimates from spectroscopy of cesium activity.")
GERMAN = ("Der Mieter zahlt die Miete monatlich an den Vermieter. Die Wohnung und der "
          "Stellplatz sind in der Miete enthalten und der Vermieter ist für die Reparaturen "
          "zuständig, wenn das Gebäude Schäden hat. Die Kaution wird nach der Übergabe der "
          "Wohnung zurückgezahlt, sofern keine Mängel vorliegen und alle Schlüssel da sind.")
GERMAN_LEASE_WORDS = GERMAN


def _page(i, text, kind=BlockKind.PARAGRAPH):
    text = text if kind != BlockKind.PARAGRAPH else f"{text} {text}"  # enough words to compare
    return [Block(id=f"{i}", page=i, kind=kind, text=text, source=SourceKind.NATIVE)]


def _ref(i):
    return PageRef(packet_id="p", index=i)


def test_content_words_skip_furniture_short_words_and_numbers():
    blocks = _page(0, "The tenant pays 1200 rent") + [
        Block(id="h", page=0, kind=BlockKind.HEADER, text="Running Header Words", source=SourceKind.NATIVE)]
    words = content_words(blocks)
    assert "tenant" in words and "rent" in words
    assert "running" not in words and "1200" not in words and "the" not in words


def test_similarity_is_high_within_a_topic_and_low_across():
    a, b, c = content_words(_page(0, LEASE)), content_words(_page(1, LEASE2)), content_words(_page(2, REACTOR))
    assert topic_similarity(a, b) > 0.1
    assert topic_similarity(a, c) < 0.01


def test_language_id_is_conservative():
    assert detect_language(LEASE) == "en"
    assert detect_language(GERMAN) == "de"
    assert detect_language("Lease 2003 page 4") is None  # too little text to tell
    assert detect_language("賃貸借契約書 第一条 本契約は、賃貸人と賃借人の間で締結される。" * 3) == "ja"


def test_topic_shift_alone_is_too_weak_to_split():
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, LEASE))
    d.observe(_ref(1), _page(1, LEASE2))
    assert d.observe(_ref(2), _page(2, REACTOR)) is None


def test_topic_shift_combines_with_a_title():
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, LEASE))
    d.observe(_ref(1), _page(1, LEASE2))
    sig = d.observe(_ref(2), _page(2, "Reactor Physics Report", BlockKind.TITLE) + _page(2, REACTOR))
    assert sig is not None and "topic_shift" in sig.reason
    plain = BoundaryDetector()
    plain.observe(_ref(0), _page(0, LEASE))
    plain.observe(_ref(1), _page(1, LEASE2))
    same = plain.observe(_ref(2), _page(2, "Parking Rules", BlockKind.TITLE) + _page(2, LEASE))
    assert same is None or "topic_shift" not in same.reason
    assert sig.confidence > (same.confidence if same else 0)


def test_no_topic_shift_on_sparse_pages():
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, "Figure 3. Rent schedule"))
    sig = d.observe(_ref(1), _page(1, "Reactor Physics Report", BlockKind.TITLE) + _page(1, REACTOR))
    assert sig is None or "topic_shift" not in sig.reason


def test_language_change_is_a_signal():
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, LEASE))
    sig = d.observe(_ref(1), _page(1, "Mietvertrag", BlockKind.TITLE) + _page(1, GERMAN))
    assert sig is not None and "language_changed" in sig.reason


def test_topic_shift_needs_prose_on_both_pages():
    # A table page (no sentences, so no language) has non-topical vocabulary.
    table = " ".join(f"Deposits Loans Securities Borrowings Interest Maturity Balance {i}" for i in range(12))
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, table, BlockKind.TABLE))
    sig = d.observe(_ref(1), _page(1, "Reactor Physics Report", BlockKind.TITLE) + _page(1, REACTOR))
    assert sig is None or "topic_shift" not in sig.reason


def test_language_change_alone_does_not_split():
    # A language change also zeroes the vocabulary overlap; the two must count once.
    d = BoundaryDetector()
    d.observe(_ref(0), _page(0, LEASE))
    d.observe(_ref(1), _page(1, LEASE2))
    sig = d.observe(_ref(2), _page(2, GERMAN + " " + GERMAN))
    assert sig is None


def test_kana_sparse_japanese_is_not_a_language_change():
    from packet.topic import same_language

    assert same_language("ja", "han") and same_language("han", "ja")
    assert not same_language("en", "de")
