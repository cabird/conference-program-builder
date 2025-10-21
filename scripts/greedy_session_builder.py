"""
Greedy Session Builder for Conference Program Creation

This module implements a fast greedy algorithm for assigning papers to conference sessions.
It uses First-Fit-Decreasing (FFD) bin packing with topical cohesion optimization.

Algorithm Phases:
  0. Prep and partition papers into primary/secondary pools
  1. Primary-only session seeding (high-volume topics first)
  2. Top-up with secondary matches
  3. Mixed sessions from leftovers
  4. Local search optimization (relocations and swaps)
  5. Finalization and export

Author: Generated for Conference Program Creation Pipeline
"""

import json
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


def get_paper_tag(paper: Dict, tag_type: str) -> Optional[str]:
    """
    Get a tag from a paper, handling both nested (tags.primary_tag) and root level.

    Args:
        paper: Paper dictionary
        tag_type: 'primary_tag', 'secondary_tag', or 'tertiary_tag'

    Returns:
        The tag value, or None if not found
    """
    # Try root level first
    if tag_type in paper:
        return paper[tag_type]

    # Try nested in tags
    if isinstance(paper.get('tags'), dict) and tag_type in paper['tags']:
        return paper['tags'][tag_type]

    return None


class Session:
    """Represents a conference session with papers and capacity constraints."""

    def __init__(self, session_id: str, topic: str, capacity: int):
        """
        Initialize a session.

        Args:
            session_id: Unique identifier for the session
            topic: Primary topic/theme for the session
            capacity: Maximum duration in minutes
        """
        self.id = session_id
        self.topics = [topic]  # List to allow 2-topic extension
        self.papers = []  # List of {'paper': paper_obj, 'match': 'primary'|'secondary'|'none'}
        self.capacity = capacity
        self.minutes_used = 0

    def can_add(self, paper: Dict) -> bool:
        """Check if a paper fits in this session."""
        return self.minutes_used + paper['minutes'] <= self.capacity

    def add_paper(self, paper: Dict, match_type: str = 'primary') -> None:
        """
        Add a paper to this session.

        Args:
            paper: Paper dictionary with id, minutes, tags, etc.
            match_type: 'primary', 'secondary', 'tertiary', or 'none'
        """
        self.papers.append({'paper': paper, 'match': match_type})
        self.minutes_used += paper['minutes']

    def remove_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Remove a paper from this session.

        Args:
            paper_id: ID of paper to remove

        Returns:
            The removed paper dictionary, or None if not found
        """
        for i, entry in enumerate(self.papers):
            if entry['paper']['id'] == paper_id:
                removed = self.papers.pop(i)
                self.minutes_used -= removed['paper']['minutes']
                return removed['paper']
        return None

    def compute_score(self, weights: Dict[str, float]) -> float:
        """
        Compute the quality score for this session.

        Score = utilization_weight * utilization +
                primary_weight * primary_cohesion +
                secondary_weight * secondary_cohesion

        Args:
            weights: Dictionary with 'utilization', 'primary', 'secondary' keys

        Returns:
            Weighted score for this session
        """
        # Utilization component
        util = self.minutes_used / self.capacity if self.capacity > 0 else 0

        # Cohesion components
        n = len(self.papers)
        if n == 0:
            return 0

        primary_matches = sum(1 for p in self.papers
                             if get_paper_tag(p['paper'], 'primary_tag') in self.topics)
        secondary_matches = sum(1 for p in self.papers
                               if get_paper_tag(p['paper'], 'secondary_tag') in self.topics)
        tertiary_matches = sum(1 for p in self.papers
                              if get_paper_tag(p['paper'], 'tertiary_tag') in self.topics)

        primary_cohesion = primary_matches / n
        secondary_cohesion = secondary_matches / n
        tertiary_cohesion = tertiary_matches / n

        return (weights.get('utilization', 1.0) * util +
                weights.get('primary', 2.0) * primary_cohesion +
                weights.get('secondary', 1.0) * secondary_cohesion +
                weights.get('tertiary', 0.5) * tertiary_cohesion)

    @property
    def remaining(self) -> int:
        """Get remaining capacity in minutes."""
        return self.capacity - self.minutes_used

    def to_dict(self) -> Dict:
        """Export session as a dictionary."""
        return {
            'session_id': self.id,
            'topics': self.topics,
            'minutes_used': self.minutes_used,
            'remaining': self.remaining,
            'total_minutes': self.minutes_used,  # For compatibility with session_analysis.py
            'unused_minutes': self.remaining,     # For compatibility with session_analysis.py
            'utilization': self.minutes_used / self.capacity if self.capacity > 0 else 0,
            'papers': [
                {
                    'id': entry['paper']['id'],
                    'title': entry['paper'].get('title', ''),
                    'track': entry['paper'].get('track', ''),
                    'minutes': entry['paper']['minutes'],
                    'match': entry['match']
                }
                for entry in self.papers
            ]
        }


class GreedySessionBuilder:
    """
    Greedy algorithm for assigning papers to conference sessions.

    Uses First-Fit-Decreasing (FFD) bin packing with topical cohesion optimization
    and local search improvements.
    """

    def __init__(self, papers: List[Dict], config: Dict):
        """
        Initialize the greedy session builder.

        Args:
            papers: List of paper dictionaries with id, track, minutes, primary_tag, etc.
            config: Configuration dictionary with session_duration, min_fill_ratio, etc.
        """
        self.papers = papers
        self.config = config

        # Initialize tracking structures
        self.sessions: List[Session] = []
        self.assigned: Set[str] = set()
        self.unassigned: Set[str] = set(p['id'] for p in papers)

        # Build primary and secondary pools
        self._build_pools()

    def _build_pools(self) -> None:
        """Build and sort primary/secondary/tertiary topic pools using FFD strategy."""
        self.primary_pools: Dict[str, List[Dict]] = defaultdict(list)
        self.secondary_pools: Dict[str, List[Dict]] = defaultdict(list)
        self.tertiary_pools: Dict[str, List[Dict]] = defaultdict(list)

        for paper in self.papers:
            # Add to primary pool
            primary_tag = get_paper_tag(paper, 'primary_tag')
            if primary_tag:
                self.primary_pools[primary_tag].append(paper)

            # Add to secondary pool
            secondary_tag = get_paper_tag(paper, 'secondary_tag')
            if secondary_tag:
                self.secondary_pools[secondary_tag].append(paper)

            # Add to tertiary pool
            tertiary_tag = get_paper_tag(paper, 'tertiary_tag')
            if tertiary_tag:
                self.tertiary_pools[tertiary_tag].append(paper)

        # Sort each pool by minutes descending (First-Fit-Decreasing)
        for topic in self.primary_pools:
            self.primary_pools[topic].sort(key=lambda p: p['minutes'], reverse=True)
        for topic in self.secondary_pools:
            self.secondary_pools[topic].sort(key=lambda p: p['minutes'], reverse=True)
        for topic in self.tertiary_pools:
            self.tertiary_pools[topic].sort(key=lambda p: p['minutes'], reverse=True)

    def build(self) -> List[Session]:
        """
        Execute the complete greedy session building algorithm.

        Returns:
            List of Session objects with assigned papers
        """
        print("Starting greedy session builder...")

        # Phase 1: Primary-only session seeding
        print("\nPhase 1: Primary-only session seeding")
        self._phase1_primary_seeding()
        print(f"  Created {len(self.sessions)} sessions")
        print(f"  Assigned {len(self.assigned)} papers")

        # Phase 2: Top-up with secondary matches
        print("\nPhase 2: Top-up with secondary matches")
        assigned_before = len(self.assigned)
        self._phase2_secondary_topup()
        print(f"  Added {len(self.assigned) - assigned_before} papers via secondary matches")

        # Phase 2.5: Top-up with tertiary matches
        print("\nPhase 2.5: Top-up with tertiary matches")
        assigned_before = len(self.assigned)
        self._phase2_5_tertiary_topup()
        print(f"  Added {len(self.assigned) - assigned_before} papers via tertiary matches")

        # Phase 3: Mixed sessions from leftovers
        print("\nPhase 3: Mixed sessions from leftovers")
        sessions_before = len(self.sessions)
        self._phase3_mixed_sessions()
        print(f"  Created {len(self.sessions) - sessions_before} mixed sessions")
        print(f"  Assigned {len(self.assigned)} total papers")

        # Phase 4: Local search optimization
        print("\nPhase 4: Local search optimization")
        self._phase4_local_search()

        # Phase 4.5: Two-topic sessions for remaining papers (after optimization)
        # Only run if enabled in config
        if self.config.get('allow_two_topic_sessions', True):
            print("\nPhase 4.5: Two-topic sessions for remaining papers")
            sessions_before = len(self.sessions)
            assigned_before = len(self.assigned)
            self._phase4_5_two_topic_sessions()
            print(f"  Created {len(self.sessions) - sessions_before} two-topic sessions")
            print(f"  Assigned {len(self.assigned) - assigned_before} papers via two-topic sessions")
        else:
            print("\nPhase 4.5: Skipped (two-topic sessions disabled)")

        # Phase 5: Compute final metrics
        print("\nPhase 5: Computing final metrics")
        self._compute_metrics()

        return self.sessions

    def _phase1_primary_seeding(self) -> None:
        """Phase 1: Create sessions for high-volume topics using primary matches only."""
        # Sort topics by total primary minutes (descending)
        topic_volumes = {
            topic: sum(p['minutes'] for p in papers if p['id'] not in self.assigned)
            for topic, papers in self.primary_pools.items()
        }
        sorted_topics = sorted(topic_volumes.items(), key=lambda x: x[1], reverse=True)

        min_fill_ratio = self.config.get('min_fill_ratio', 0.75)
        session_duration = self.config['session_duration']

        for topic, volume in sorted_topics:
            while volume >= min_fill_ratio * session_duration:
                session = Session(
                    f"S{len(self.sessions)+1:02d}",
                    topic,
                    session_duration
                )

                # Greedy pack from primary pool (FFD - already sorted)
                papers_to_pack = [p for p in self.primary_pools[topic]
                                if p['id'] not in self.assigned]

                for paper in papers_to_pack:
                    if session.can_add(paper):
                        session.add_paper(paper, 'primary')
                        self.assigned.add(paper['id'])
                        self.unassigned.remove(paper['id'])

                # Check MIN_FILL_RATIO before keeping session
                if session.minutes_used >= min_fill_ratio * session_duration:
                    self.sessions.append(session)
                else:
                    # Disband and return papers to unassigned
                    for entry in session.papers:
                        p = entry['paper']
                        self.assigned.remove(p['id'])
                        self.unassigned.add(p['id'])
                    break

                # Recalculate volume for next iteration
                volume = sum(p['minutes'] for p in self.primary_pools[topic]
                           if p['id'] not in self.assigned)

    def _phase2_secondary_topup(self) -> None:
        """Phase 2: Fill remaining capacity in existing sessions with secondary matches."""
        # Sort sessions by remaining capacity (fill nearly-full sessions first)
        sorted_sessions = sorted(self.sessions, key=lambda s: s.remaining)

        for session in sorted_sessions:
            if session.remaining == 0:
                continue

            # Get secondary candidates for this session's topic
            topic = session.topics[0]
            candidates = [p for p in self.secondary_pools.get(topic, [])
                         if p['id'] not in self.assigned]

            for paper in candidates:
                if session.can_add(paper):
                    session.add_paper(paper, 'secondary')
                    self.assigned.add(paper['id'])
                    self.unassigned.remove(paper['id'])

    def _phase2_5_tertiary_topup(self) -> None:
        """Phase 2.5: Fill remaining capacity in existing sessions with tertiary matches."""
        # Sort sessions by remaining capacity (fill nearly-full sessions first)
        sorted_sessions = sorted(self.sessions, key=lambda s: s.remaining)

        for session in sorted_sessions:
            if session.remaining == 0:
                continue

            # Get tertiary candidates for this session's topic
            topic = session.topics[0]
            candidates = [p for p in self.tertiary_pools.get(topic, [])
                         if p['id'] not in self.assigned]

            for paper in candidates:
                if session.can_add(paper):
                    session.add_paper(paper, 'tertiary')
                    self.assigned.add(paper['id'])
                    self.unassigned.remove(paper['id'])

    def _phase3_mixed_sessions(self) -> None:
        """Phase 3: Build sessions from remaining unassigned papers."""
        unassigned_list = sorted(
            [p for p in self.papers if p['id'] in self.unassigned],
            key=lambda p: p['minutes'],
            reverse=True
        )

        min_fill_ratio = self.config.get('min_fill_ratio', 0.75)
        session_duration = self.config['session_duration']
        papers_to_process = list(unassigned_list)  # Work on a copy

        while papers_to_process:
            paper = papers_to_process.pop(0)
            if paper['id'] not in self.unassigned:  # Already assigned
                continue

            # Start new session with this paper's primary topic
            session = Session(
                f"S{len(self.sessions)+1:02d}",
                get_paper_tag(paper, 'primary_tag') or 'Mixed',
                session_duration
            )

            # Check if paper fits (don't create session if it's too large)
            if not session.can_add(paper):
                continue  # Skip this paper, it's too large for any session

            session.add_paper(paper, 'primary')
            self.assigned.add(paper['id'])
            self.unassigned.remove(paper['id'])

            # Fill session greedily from remaining papers
            # Prefer papers with at least tertiary match to avoid "no match" papers
            allow_no_match = self.config.get('allow_no_match_in_mixed', False)
            candidates = [p for p in papers_to_process if p['id'] in self.unassigned]

            for candidate in candidates:
                if session.can_add(candidate):
                    # Determine match type
                    match_type = 'none'
                    if get_paper_tag(candidate, 'primary_tag') in session.topics:
                        match_type = 'primary'
                    elif get_paper_tag(candidate, 'secondary_tag') in session.topics:
                        match_type = 'secondary'
                    elif get_paper_tag(candidate, 'tertiary_tag') in session.topics:
                        match_type = 'tertiary'

                    # Skip papers with no match unless explicitly allowed
                    if match_type == 'none' and not allow_no_match:
                        continue

                    session.add_paper(candidate, match_type)
                    self.assigned.add(candidate['id'])
                    self.unassigned.remove(candidate['id'])

            # Check MIN_FILL_RATIO before keeping
            if session.minutes_used >= min_fill_ratio * session_duration:
                self.sessions.append(session)
            else:
                # Return papers to unassigned if can't make threshold
                for entry in session.papers:
                    p = entry['paper']
                    self.assigned.remove(p['id'])
                    self.unassigned.add(p['id'])
                break  # Can't make more sessions

    def _phase4_local_search(self) -> None:
        """Phase 4: Multi-pass local search with relocations and swaps."""
        start_time = time.time()
        swap_passes = self.config.get('swap_passes', 3)
        time_budget = self.config.get('time_budget_seconds', 5)

        initial_score = self._get_global_score()
        initial_unassigned = len(self.unassigned)
        print(f"  Initial score: {initial_score:.3f}, unassigned: {initial_unassigned}")

        for pass_num in range(swap_passes):
            if time.time() - start_time > time_budget:
                print(f"  Time budget exceeded, stopping at pass {pass_num}")
                break

            improved = False

            # Try relocations first (simpler and often effective)
            if self._relocation_pass():
                improved = True
                print(f"  Pass {pass_num+1}: Relocation improved score")

            # Try swaps
            if self._swap_pass():
                improved = True
                print(f"  Pass {pass_num+1}: Swap improved score")

            # Always try to assign leftover papers via swap-in (even if other passes didn't improve)
            if self.unassigned and (time.time() - start_time < time_budget):
                if self._leftover_swapin_pass(start_time, time_budget):
                    improved = True
                    print(f"  Pass {pass_num+1}: Swap-in placed leftover papers")

            # Only stop if no improvements AND no unassigned papers left
            if not improved and not self.unassigned:
                print(f"  Pass {pass_num+1}: No improvement and all papers assigned")
                break
            elif not improved:
                print(f"  Pass {pass_num+1}: No improvement this pass, continuing...")
                # Continue to next pass to try placing more leftovers

        final_score = self._get_global_score()
        final_unassigned = len(self.unassigned)
        print(f"  Final score: {final_score:.3f} (improvement: {final_score - initial_score:.3f})")
        print(f"  Placed {initial_unassigned - final_unassigned} leftover papers")

    def _get_global_score(self) -> float:
        """Compute the global score across all sessions."""
        weights = self.config.get('weights', {
            'utilization': 1.0,
            'primary': 2.0,
            'secondary': 1.0
        })
        return sum(s.compute_score(weights) for s in self.sessions)

    def _relocation_pass(self) -> bool:
        """Try moving papers to better sessions. Returns True if any improvement made."""
        current_score = self._get_global_score()
        weights = self.config.get('weights', {})

        for session in self.sessions:
            # Identify weak-fit papers (candidates for relocation)
            weak_papers = [entry for entry in session.papers
                          if entry['match'] != 'primary']

            for entry in weak_papers:
                paper = entry['paper']

                # Try moving to other sessions
                for target_session in self.sessions:
                    if target_session.id == session.id:
                        continue

                    # Check if move is valid
                    if target_session.can_add(paper):
                        # Evaluate move
                        session.remove_paper(paper['id'])

                        # Determine match type in target session
                        match_type = 'none'
                        if get_paper_tag(paper, 'primary_tag') in target_session.topics:
                            match_type = 'primary'
                        elif get_paper_tag(paper, 'secondary_tag') in target_session.topics:
                            match_type = 'secondary'

                        target_session.add_paper(paper, match_type)

                        new_score = self._get_global_score()

                        if new_score > current_score:
                            # Keep the move (first-improving)
                            return True
                        else:
                            # Undo move
                            target_session.remove_paper(paper['id'])
                            session.add_paper(paper, entry['match'])

        return False

    def _swap_pass(self) -> bool:
        """Try swapping papers between sessions. Returns True if any improvement made."""
        current_score = self._get_global_score()

        # Build candidate lists (avoid O(N²))
        candidates = []
        for session in self.sessions:
            weak_papers = [entry for entry in session.papers
                          if entry['match'] != 'primary'][:10]  # Top 10 candidates per session
            candidates.extend([(session, entry) for entry in weak_papers])

        for sess_i, entry_i in candidates:
            paper_i = entry_i['paper']

            for sess_j, entry_j in candidates:
                if sess_i.id == sess_j.id:
                    continue

                paper_j = entry_j['paper']

                # Check if swap is feasible (both papers must fit)
                if (sess_i.remaining + paper_i['minutes'] >= paper_j['minutes'] and
                    sess_j.remaining + paper_j['minutes'] >= paper_i['minutes']):

                    # Try swap
                    sess_i.remove_paper(paper_i['id'])
                    sess_j.remove_paper(paper_j['id'])

                    # Determine match types after swap
                    match_i = 'none'
                    if get_paper_tag(paper_i, 'primary_tag') in sess_j.topics:
                        match_i = 'primary'
                    elif get_paper_tag(paper_i, 'secondary_tag') in sess_j.topics:
                        match_i = 'secondary'

                    match_j = 'none'
                    if get_paper_tag(paper_j, 'primary_tag') in sess_i.topics:
                        match_j = 'primary'
                    elif get_paper_tag(paper_j, 'secondary_tag') in sess_i.topics:
                        match_j = 'secondary'

                    sess_j.add_paper(paper_i, match_i)
                    sess_i.add_paper(paper_j, match_j)

                    new_score = self._get_global_score()

                    if new_score > current_score:
                        # Keep the swap (first-improving)
                        return True
                    else:
                        # Undo swap
                        sess_j.remove_paper(paper_i['id'])
                        sess_i.remove_paper(paper_j['id'])
                        sess_i.add_paper(paper_i, entry_i['match'])
                        sess_j.add_paper(paper_j, entry_j['match'])

        return False

    def _leftover_swapin_pass(self, start_time: float, time_budget: float) -> bool:
        """
        Try to place leftover papers by swapping them with weaker papers in sessions.

        Strategy:
        1. For each leftover paper, find sessions where it has a tag match
        2. Try to swap it with weak-fit papers in those sessions
        3. Accept swaps that improve global score or maintain score while reducing leftovers

        Returns True if any leftovers were successfully placed.
        """
        import time as time_module

        if not self.unassigned:
            return False

        placed_any = False
        leftover_papers = [p for p in self.papers if p['id'] in self.unassigned]

        # Sort leftovers by best potential match quality
        # Prioritize papers that have primary/secondary matches available
        def get_leftover_priority(paper):
            primary_tag = get_paper_tag(paper, 'primary_tag')
            secondary_tag = get_paper_tag(paper, 'secondary_tag')
            tertiary_tag = get_paper_tag(paper, 'tertiary_tag')

            # Count sessions where this paper could match
            primary_sessions = sum(1 for s in self.sessions if primary_tag in s.topics)
            secondary_sessions = sum(1 for s in self.sessions if secondary_tag in s.topics)
            tertiary_sessions = sum(1 for s in self.sessions if tertiary_tag in s.topics)

            return (primary_sessions * 100 + secondary_sessions * 10 + tertiary_sessions, -paper['minutes'])

        leftover_papers.sort(key=get_leftover_priority, reverse=True)

        for leftover_paper in leftover_papers:
            if time_module.time() - start_time > time_budget:
                break

            if leftover_paper['id'] not in self.unassigned:
                continue  # Already placed

            # Find sessions where this paper has a tag match
            primary_tag = get_paper_tag(leftover_paper, 'primary_tag')
            secondary_tag = get_paper_tag(leftover_paper, 'secondary_tag')
            tertiary_tag = get_paper_tag(leftover_paper, 'tertiary_tag')

            candidate_sessions = []
            for session in self.sessions:
                if primary_tag in session.topics:
                    candidate_sessions.append((session, 'primary', 100))
                elif secondary_tag in session.topics:
                    candidate_sessions.append((session, 'secondary', 50))
                elif tertiary_tag in session.topics:
                    candidate_sessions.append((session, 'tertiary', 25))

            # Sort by match quality
            candidate_sessions.sort(key=lambda x: x[2], reverse=True)

            # Try to swap into each candidate session
            for session, match_type, priority in candidate_sessions:
                if time_module.time() - start_time > time_budget:
                    break

                # Find weak papers in this session to swap out
                weak_papers = [
                    entry for entry in session.papers
                    if entry['match'] in ['none', 'tertiary', 'secondary']
                ]

                # Sort by weakest match first
                match_priority = {'none': 0, 'tertiary': 1, 'secondary': 2, 'primary': 3}
                weak_papers.sort(key=lambda e: (match_priority[e['match']], -e['paper']['minutes']))

                for weak_entry in weak_papers:
                    weak_paper = weak_entry['paper']

                    # Check if swap is feasible (size-wise)
                    if session.remaining + weak_paper['minutes'] >= leftover_paper['minutes']:
                        # Try the swap
                        current_score = self._get_global_score()
                        current_unassigned = len(self.unassigned)

                        # Remove weak paper from session
                        session.remove_paper(weak_paper['id'])
                        self.assigned.remove(weak_paper['id'])
                        self.unassigned.add(weak_paper['id'])

                        # Add leftover paper to session
                        if session.can_add(leftover_paper):
                            session.add_paper(leftover_paper, match_type)
                            self.assigned.add(leftover_paper['id'])
                            self.unassigned.remove(leftover_paper['id'])

                            new_score = self._get_global_score()
                            new_unassigned = len(self.unassigned)

                            # Accept if score improves OR score stays same/close and we reduced unassigned
                            # Be aggressive about placing papers - allow score decrease up to tolerance
                            score_tolerance = self.config.get('leftover_swap_score_tolerance', 2.0)
                            score_acceptable = (new_score >= current_score - score_tolerance)
                            reduced_unassigned = (new_unassigned < current_unassigned)

                            if score_acceptable and reduced_unassigned:
                                # Keep the swap
                                placed_any = True
                                break  # Move to next leftover paper
                            else:
                                # Undo the swap
                                session.remove_paper(leftover_paper['id'])
                                self.assigned.remove(leftover_paper['id'])
                                self.unassigned.add(leftover_paper['id'])

                                session.add_paper(weak_paper, weak_entry['match'])
                                self.assigned.add(weak_paper['id'])
                                self.unassigned.remove(weak_paper['id'])
                        else:
                            # Undo removal of weak paper
                            session.add_paper(weak_paper, weak_entry['match'])
                            self.assigned.add(weak_paper['id'])
                            self.unassigned.remove(weak_paper['id'])

                # If we placed this leftover, move to next one
                if leftover_paper['id'] not in self.unassigned:
                    break

        return placed_any

    def _phase4_5_two_topic_sessions(self) -> None:
        """
        Phase 4.5: Create 2-topic sessions from remaining unassigned papers.

        Strategy:
        1. Group leftover papers by their primary tags
        2. For each pair of tags with papers, try to create a 2-topic session
        3. Prioritize topic pairs that share papers (via secondary/tertiary tags)
        4. Pack greedily using FFD within each 2-topic session
        """
        if not self.unassigned:
            return

        min_fill_ratio = self.config.get('min_fill_ratio', 0.70)
        session_duration = self.config['session_duration']

        # Group unassigned papers by primary tag
        papers_by_tag = defaultdict(list)
        for paper in self.papers:
            if paper['id'] in self.unassigned:
                primary_tag = get_paper_tag(paper, 'primary_tag')
                if primary_tag:
                    papers_by_tag[primary_tag].append(paper)

        # Sort each group by minutes descending (FFD)
        for tag in papers_by_tag:
            papers_by_tag[tag].sort(key=lambda p: p['minutes'], reverse=True)

        # Find good topic pairs (topics that complement each other)
        topic_pairs = []
        tags = list(papers_by_tag.keys())

        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                # Calculate affinity between these tags
                # Papers that have tag1 as primary and tag2 as secondary/tertiary (or vice versa)
                affinity = 0
                papers1 = papers_by_tag[tag1]
                papers2 = papers_by_tag[tag2]

                for p in papers1:
                    if get_paper_tag(p, 'secondary_tag') == tag2 or get_paper_tag(p, 'tertiary_tag') == tag2:
                        affinity += 10

                for p in papers2:
                    if get_paper_tag(p, 'secondary_tag') == tag1 or get_paper_tag(p, 'tertiary_tag') == tag1:
                        affinity += 10

                # Also consider total volume
                total_minutes = sum(p['minutes'] for p in papers1) + sum(p['minutes'] for p in papers2)

                topic_pairs.append((tag1, tag2, affinity, total_minutes))

        # Sort by affinity first, then by volume
        topic_pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)

        # Create 2-topic sessions
        used_tags = set()

        for tag1, tag2, affinity, volume in topic_pairs:
            # Skip if either tag already used in a 2-topic session or has no papers left
            if tag1 in used_tags or tag2 in used_tags:
                continue

            papers_available = [p for p in papers_by_tag[tag1] + papers_by_tag[tag2]
                               if p['id'] in self.unassigned]

            if not papers_available:
                continue

            # Check if we have enough volume for a session
            total_available = sum(p['minutes'] for p in papers_available)
            if total_available < min_fill_ratio * session_duration:
                continue

            # Create 2-topic session
            session = Session(
                f"S{len(self.sessions)+1:02d}",
                tag1,  # Primary topic
                session_duration
            )
            session.topics.append(tag2)  # Add second topic

            # Pack papers greedily (FFD - already sorted)
            for paper in papers_available:
                if paper['id'] not in self.unassigned:
                    continue

                if session.can_add(paper):
                    # Determine match type
                    primary_tag = get_paper_tag(paper, 'primary_tag')
                    secondary_tag = get_paper_tag(paper, 'secondary_tag')
                    tertiary_tag = get_paper_tag(paper, 'tertiary_tag')

                    if primary_tag in session.topics:
                        match_type = 'primary'
                    elif secondary_tag in session.topics:
                        match_type = 'secondary'
                    elif tertiary_tag in session.topics:
                        match_type = 'tertiary'
                    else:
                        match_type = 'none'

                    session.add_paper(paper, match_type)
                    self.assigned.add(paper['id'])
                    self.unassigned.remove(paper['id'])

            # Check if session meets minimum fill ratio
            if session.minutes_used >= min_fill_ratio * session_duration:
                self.sessions.append(session)
                used_tags.add(tag1)
                used_tags.add(tag2)
            else:
                # Disband session if it doesn't meet threshold
                for entry in session.papers:
                    p = entry['paper']
                    self.assigned.remove(p['id'])
                    self.unassigned.add(p['id'])

        # Final pass: try to create single-tag sessions from large remaining groups
        for tag in papers_by_tag:
            if tag in used_tags:
                continue

            papers_available = [p for p in papers_by_tag[tag] if p['id'] in self.unassigned]
            if not papers_available:
                continue

            total_available = sum(p['minutes'] for p in papers_available)
            if total_available < min_fill_ratio * session_duration:
                continue

            # Create single-topic session
            session = Session(
                f"S{len(self.sessions)+1:02d}",
                tag,
                session_duration
            )

            for paper in papers_available:
                if session.can_add(paper):
                    session.add_paper(paper, 'primary')
                    self.assigned.add(paper['id'])
                    self.unassigned.remove(paper['id'])

            if session.minutes_used >= min_fill_ratio * session_duration:
                self.sessions.append(session)

    def _compute_metrics(self) -> Dict:
        """Compute and print final metrics."""
        if not self.sessions:
            print("  No sessions created")
            return {}

        # Utilization metrics
        utilizations = [s.minutes_used / s.capacity for s in self.sessions]
        avg_util = sum(utilizations) / len(utilizations)
        median_util = sorted(utilizations)[len(utilizations) // 2]

        # Cohesion metrics
        total_papers = sum(len(s.papers) for s in self.sessions)
        primary_matches = sum(
            sum(1 for p in s.papers if get_paper_tag(p['paper'], 'primary_tag') in s.topics)
            for s in self.sessions
        )
        secondary_matches = sum(
            sum(1 for p in s.papers if get_paper_tag(p['paper'], 'secondary_tag') in s.topics)
            for s in self.sessions
        )

        primary_rate = primary_matches / total_papers if total_papers > 0 else 0
        secondary_rate = secondary_matches / total_papers if total_papers > 0 else 0

        # Leftover papers
        leftover_count = len(self.unassigned)

        print(f"  Sessions created: {len(self.sessions)}")
        print(f"  Papers assigned: {len(self.assigned)}")
        print(f"  Papers unassigned: {leftover_count}")
        print(f"  Average utilization: {avg_util:.1%}")
        print(f"  Median utilization: {median_util:.1%}")
        print(f"  Primary match rate: {primary_rate:.1%}")
        print(f"  Secondary match rate: {secondary_rate:.1%}")

        self.metrics = {
            'sessions_created': len(self.sessions),
            'papers_assigned': len(self.assigned),
            'papers_unassigned': leftover_count,
            'avg_utilization': avg_util,
            'median_utilization': median_util,
            'primary_match_rate': primary_rate,
            'secondary_match_rate': secondary_rate
        }

        return self.metrics

    def export_sessions(self) -> Dict:
        """Export sessions and metrics as a dictionary."""
        return {
            'sessions': [s.to_dict() for s in self.sessions],
            'leftovers': [
                {
                    'id': p['id'],
                    'title': p.get('title', ''),
                    'track': p.get('track', ''),
                    'minutes': p['minutes'],
                    'primary_tag': get_paper_tag(p, 'primary_tag') or '',
                    'secondary_tag': get_paper_tag(p, 'secondary_tag') or ''
                }
                for p in self.papers if p['id'] in self.unassigned
            ],
            'metrics': getattr(self, 'metrics', {})
        }
