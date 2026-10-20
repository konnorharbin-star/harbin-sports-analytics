import copy
import hashlib
import json

import pytest

from harbin.official_personnel import evaluate_article, publisher

NOW = "2026-10-20T12:00:00Z"


def game():
    return {"game_id": "123", "season": 2026, "week": 8, "date": "2026-10-24T17:00:00Z",
            "home_id": "1", "away_id": "2", "home_team": "Alabama", "away_team": "Georgia"}


def roster():
    return [{"team_id": 1, "athlete_id": 7, "full_name": "Example Player", "position_abbreviation": "QB"},
            {"team_id": 2, "athlete_id": 8, "full_name": "Other Player", "position_abbreviation": "QB"}]


def article(body="Example Player will start against Georgia in Week 8.", **changes):
    metadata = {"@type": "NewsArticle", "datePublished": "2026-10-20T11:00:00Z",
                "headline": "Alabama vs Georgia: Week 8 starter news", "articleBody": body, **changes}
    html = '<script type="application/ld+json">' + json.dumps(metadata) + '</script>'
    return {"html": html, "snapshot_sha256": hashlib.sha256(html.encode()).hexdigest(),
            "captured_utc": NOW, "source_url": "https://www.secsports.com/news/test"}


def test_explicit_official_starter_binds_to_unique_team_roster_id():
    result = evaluate_article(article(), game(), roster(), as_of=NOW)
    assert result["rejections"] == []
    assert result["starters"]["1"]["athlete_id"] == "7"


@pytest.mark.parametrize("body", [
    "Example Player is expected to start in Week 8.",
    "Example Player will start if healthy in Week 8.",
    "Example Player will start next week after Week 8.",
    "Example Player will not start in Week 8.",
    "Example Player will start in Week 7.",
])
def test_conditional_negative_and_other_week_statements_never_confirm(body):
    assert not evaluate_article(article(body), game(), roster(), as_of=NOW)["starters"]


@pytest.mark.parametrize("changes", [
    {"headline": "Alabama vs Georgia Tech: Week 8 starter news"},
    {"datePublished": "2026-10-25T12:00:00Z"},
    {"datePublished": "2026-10-20"},
    {"dateModified": "2026-10-21T12:00:00Z"},
    {"articleBody": None},
])
def test_wrong_matchups_and_bad_source_clocks_remain_unknown(changes):
    result = evaluate_article(article(**changes), game(), roster(), as_of=NOW)
    assert not result["starters"]
    assert result["rejections"]


def test_out_declaration_vetoes_same_article_starter():
    result = evaluate_article(article("Example Player will start in Week 8. Example Player has been ruled out."), game(), roster(), as_of=NOW)
    assert not result["starters"]
    assert result["player_designations"][0]["designation"] == "OUT"


def test_ambiguous_same_name_in_two_rosters_is_not_bound():
    players = roster() + [{"team_id": 2, "athlete_id": 9, "full_name": "Example Player", "position_abbreviation": "QB"}]
    assert not evaluate_article(article(), game(), players, as_of=NOW)["starters"]


def test_untrusted_lookalike_domains_and_mutated_payload_rejected():
    assert publisher("https://secsports.com.attacker.invalid/news/") is None
    assert publisher("https://attacker@secsports.com/news/") is None
    assert publisher("https://www.secsports.com/news/") == "secsports.com"
    page = article()
    page["html"] += "changed"
    assert evaluate_article(page, game(), roster(), as_of=NOW)["rejections"] == ["OFFICIAL_ARTICLE_HASH_MISMATCH"]


def test_omission_does_not_create_active_designation():
    result = evaluate_article(article(), game(), roster(), as_of=NOW)
    assert not result["player_designations"]
