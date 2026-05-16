from places_attr_conflation.weak_labels import weak_label


def test_supports_authoritative_match():
    result = weak_label(
        candidate_value='https://example.com',
        extracted_value='https://example.com',
        source_type='official_site',
    )
    assert result.label == 'supports'
    assert result.confidence >= 0.9


def test_contradicts_mismatch():
    result = weak_label(
        candidate_value='https://example.com',
        extracted_value='https://other.com',
        source_type='aggregator',
    )
    assert result.label == 'contradicts'


def test_stale_marker_detection():
    result = weak_label(
        candidate_value='123',
        extracted_value='123',
        source_type='official_site',
        page_text='This location is permanently closed',
    )
    assert result.label == 'stale'
