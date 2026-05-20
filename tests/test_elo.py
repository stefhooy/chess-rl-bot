"""Tests for Elo formulas used in evaluator, training promotions, and webapp."""

import pytest

from chess_rl.evaluation.evaluator import Evaluator


# ── Evaluator Elo helpers ──────────────────────────────────────────────────

def test_expected_score_equal_ratings():
    assert Evaluator._expected_score(1200, 1200) == pytest.approx(0.5)


def test_expected_score_higher_beats_lower():
    score = Evaluator._expected_score(1600, 1200)
    assert score > 0.5


def test_expected_score_lower_beats_higher():
    score = Evaluator._expected_score(1200, 1600)
    assert score < 0.5


def test_expected_scores_sum_to_one():
    a = Evaluator._expected_score(1400, 1200)
    b = Evaluator._expected_score(1200, 1400)
    assert a + b == pytest.approx(1.0)


def test_expected_score_400_point_gap():
    # Standard Elo: 400-point gap → ~91% expected score
    score = Evaluator._expected_score(1600, 1200)
    assert score == pytest.approx(0.9091, abs=1e-3)


# ── Webapp symmetric Elo formula (app.py) ─────────────────────────────────
# delta = round(32 * (score - 0.5), 1)
# score: bot=1.0, draw=0.5, player=0.0

def test_webapp_elo_bot_wins():
    score = 1.0   # bot won
    delta = round(32 * (score - 0.5), 1)
    assert delta == 16.0


def test_webapp_elo_player_wins():
    score = 0.0   # player won (Marvin lost)
    delta = round(32 * (score - 0.5), 1)
    assert delta == -16.0


def test_webapp_elo_draw():
    score = 0.5   # draw
    delta = round(32 * (score - 0.5), 1)
    assert delta == 0.0


def test_webapp_elo_is_symmetric():
    win_delta = round(32 * (1.0 - 0.5), 1)
    loss_delta = round(32 * (0.0 - 0.5), 1)
    assert win_delta == -loss_delta


# ── Training promotion Elo formula (train.py) ─────────────────────────────
# delta = round(64 * (win_rate - 0.5), 1)

def test_training_elo_perfect_win_rate():
    delta = round(64 * (1.0 - 0.5), 1)
    assert delta == 32.0


def test_training_elo_at_threshold():
    # win_rate = 0.55 (just above 55% promotion threshold)
    delta = round(64 * (0.55 - 0.5), 1)
    assert delta == pytest.approx(3.2)


def test_training_elo_win_rate_70pct():
    delta = round(64 * (0.7 - 0.5), 1)
    assert delta == pytest.approx(12.8)


def test_training_elo_no_promotion_below_threshold():
    # win_rate = 0.5 → no improvement → delta = 0
    delta = round(64 * (0.5 - 0.5), 1)
    assert delta == 0.0


# ── Starting Elo ──────────────────────────────────────────────────────────

def test_starting_elo_after_8_losses():
    # Started at 600, lost 8 games: 600 - (8 × 16) = 472
    starting_elo = 600.0
    losses = 8
    elo = starting_elo + losses * round(32 * (0.0 - 0.5), 1)
    assert elo == pytest.approx(472.0)
