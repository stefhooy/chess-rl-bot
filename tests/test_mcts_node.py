"""Tests for MCTS Node — PUCT scoring, backup propagation, expansion."""

import pytest

from chess_rl.mcts.node import Node


def test_initial_state():
    node = Node(prior=0.5)
    assert node.N == 0
    assert node.W == 0.0
    assert node.P == 0.5
    assert node.Q == 0.0
    assert not node.is_expanded
    assert node.parent is None
    assert node.action is None


def test_q_is_zero_before_any_visit():
    node = Node(prior=1.0)
    assert node.Q == 0.0


def test_q_after_backup():
    node = Node(prior=1.0)
    node.backup(0.8)
    assert node.Q == pytest.approx(0.8)


def test_backup_increments_n_and_w():
    node = Node(prior=1.0)
    node.backup(0.6)
    node.backup(0.4)
    assert node.N == 2
    assert node.W == pytest.approx(1.0)
    assert node.Q == pytest.approx(0.5)


def test_backup_propagates_negated_to_parent():
    parent = Node(prior=1.0)
    child = Node(prior=0.5, parent=parent, action=42)
    parent.children[42] = child

    child.backup(0.7)

    assert child.N == 1
    assert child.W == pytest.approx(0.7)
    assert parent.N == 1
    assert parent.W == pytest.approx(-0.7)


def test_backup_alternates_sign_through_grandparent():
    grandparent = Node(prior=1.0)
    parent = Node(prior=0.5, parent=grandparent, action=0)
    child = Node(prior=0.3, parent=parent, action=1)
    grandparent.children[0] = parent
    parent.children[1] = child

    child.backup(0.5)

    assert child.W == pytest.approx(0.5)
    assert parent.W == pytest.approx(-0.5)
    assert grandparent.W == pytest.approx(0.5)


def test_expand_creates_children():
    node = Node(prior=1.0)
    assert not node.is_expanded
    node.expand({10: 0.3, 20: 0.7})
    assert node.is_expanded
    assert 10 in node.children
    assert 20 in node.children


def test_expand_sets_correct_priors():
    node = Node(prior=1.0)
    node.expand({5: 0.4, 9: 0.6})
    assert node.children[5].P == pytest.approx(0.4)
    assert node.children[9].P == pytest.approx(0.6)


def test_expand_sets_parent_reference():
    node = Node(prior=1.0)
    node.expand({3: 0.5})
    assert node.children[3].parent is node


def test_expand_does_not_overwrite_existing_child():
    node = Node(prior=1.0)
    node.expand({7: 0.9})
    original_child = node.children[7]
    node.expand({7: 0.1})  # second expand should not overwrite
    assert node.children[7] is original_child


def test_select_child_returns_highest_puct():
    parent = Node(prior=1.0)
    parent.N = 10
    parent.expand({0: 0.9, 1: 0.1})
    best = parent.select_child(c_puct=1.0)
    assert best.action == 0  # higher prior wins when both unvisited


def test_puct_score_formula():
    parent = Node(prior=1.0)
    parent.N = 4
    child = Node(prior=0.5, parent=parent, action=0)
    # U = c_puct * P * sqrt(parent.N) / (1 + N) = 2.0 * 0.5 * 2 / 1 = 2.0
    # Q = 0.0 (unvisited)
    score = child.puct_score(c_puct=2.0)
    assert score == pytest.approx(2.0)


def test_puct_decreases_after_visits():
    parent = Node(prior=1.0)
    parent.N = 9
    child = Node(prior=0.5, parent=parent, action=0)
    score_before = child.puct_score(c_puct=1.0)
    child.backup(0.0)
    parent.N += 1
    score_after = child.puct_score(c_puct=1.0)
    # U term decreases as child.N grows (denominator 1+N gets larger)
    assert score_after < score_before
