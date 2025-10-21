#!/usr/bin/env python3
"""
Fill-in Session Builder using CP-SAT constraint solver.

Takes leftover papers from greedy session creation and assigns them to new sessions
using pairwise similarity optimization with Google OR-Tools CP-SAT.
"""

from typing import Dict, List, Set, Tuple
from ortools.sat.python import cp_model
import json
from collections import defaultdict


def get_paper_tag(paper: Dict, tag_type: str) -> str:
    """Get a tag from a paper, handling both nested and root level."""
    if tag_type in paper:
        return paper[tag_type]
    if isinstance(paper.get('tags'), dict) and tag_type in paper['tags']:
        return paper['tags'][tag_type]
    return None


def calculate_similarity(paper_i: Dict, paper_j: Dict) -> int:
    """
    Calculate pairwise similarity score between two papers based on tag overlap.

    Scoring:
    - Primary-primary match: 10 points
    - Primary-secondary cross match: 6 points
    - Secondary-secondary match: 4 points
    - Tertiary overlap: 2 points

    Args:
        paper_i: First paper
        paper_j: Second paper

    Returns:
        Similarity score (0-20+)
    """
    score = 0

    # Get tags for both papers
    primary_i = get_paper_tag(paper_i, 'primary_tag')
    secondary_i = get_paper_tag(paper_i, 'secondary_tag')
    tertiary_i = get_paper_tag(paper_i, 'tertiary_tag')

    primary_j = get_paper_tag(paper_j, 'primary_tag')
    secondary_j = get_paper_tag(paper_j, 'secondary_tag')
    tertiary_j = get_paper_tag(paper_j, 'tertiary_tag')

    # Primary-primary match: 10 points
    if primary_i and primary_j and primary_i == primary_j:
        score += 10

    # Primary-secondary cross matches: 6 points each
    if primary_i and secondary_j and primary_i == secondary_j:
        score += 6
    if secondary_i and primary_j and secondary_i == primary_j:
        score += 6

    # Secondary-secondary match: 4 points
    if secondary_i and secondary_j and secondary_i == secondary_j:
        score += 4

    # Tertiary overlaps: 2 points
    tertiary_tags_i = {tertiary_i} if tertiary_i else set()
    tertiary_tags_j = {tertiary_j} if tertiary_j else set()

    # Also consider secondary as potential tertiary match
    if secondary_i:
        tertiary_tags_i.add(secondary_i)
    if secondary_j:
        tertiary_tags_j.add(secondary_j)

    if tertiary_tags_i & tertiary_tags_j:
        score += 2

    return score


class FillInSessionBuilder:
    """
    Builds sessions for leftover papers using CP-SAT constraint solver.
    """

    def __init__(self, leftover_papers: List[Dict], config: Dict):
        """
        Initialize the fill-in session builder.

        Args:
            leftover_papers: Papers that need to be assigned
            config: Session creation options from session_config.json
        """
        self.papers = leftover_papers
        self.config = config
        self.num_papers = len(leftover_papers)

        # Extract config parameters
        self.session_duration = config.get('session_duration', 90)
        self.min_fill_ratio = config.get('min_fill_ratio', 0.75)
        self.min_fill_minutes = self.session_duration * self.min_fill_ratio

        # Calculate how many sessions we might need
        total_minutes = sum(p['minutes'] for p in self.papers)
        self.max_sessions = max(1, (total_minutes + int(self.min_fill_minutes) - 1) // int(self.min_fill_minutes))

        # Pre-compute pairwise similarities
        self.similarity_matrix = self._compute_similarity_matrix()

        print(f"Fill-in session builder initialized:")
        print(f"  Leftover papers: {self.num_papers}")
        print(f"  Total minutes: {total_minutes}")
        print(f"  Session duration: {self.session_duration} min")
        print(f"  Min fill ratio: {self.min_fill_ratio:.0%} ({self.min_fill_minutes} min)")
        print(f"  Estimated sessions needed: {self.max_sessions}")

    def _compute_similarity_matrix(self) -> Dict[Tuple[int, int], int]:
        """
        Pre-compute similarity scores for all paper pairs.

        Returns:
            Dictionary mapping (i, j) -> similarity score
        """
        print("\nComputing pairwise similarity matrix...")
        similarity = {}

        for i in range(self.num_papers):
            for j in range(i + 1, self.num_papers):
                score = calculate_similarity(self.papers[i], self.papers[j])
                if score > 0:
                    similarity[(i, j)] = score

        print(f"  Found {len(similarity)} non-zero similarity pairs")
        return similarity

    def build(self) -> List[Dict]:
        """
        Build sessions using CP-SAT constraint solver.

        Returns:
            List of session dictionaries
        """
        print("\n" + "="*60)
        print("BUILDING FILL-IN SESSIONS WITH CP-SAT")
        print("="*60)

        model = cp_model.CpModel()

        # Decision variables
        print("\nCreating decision variables...")

        # x[i][j] = 1 if paper i is assigned to session j
        x = {}
        for i in range(self.num_papers):
            for j in range(self.max_sessions):
                x[i, j] = model.NewBoolVar(f'paper_{i}_in_session_{j}')

        # y[j] = 1 if session j is used
        y = {}
        for j in range(self.max_sessions):
            y[j] = model.NewBoolVar(f'session_{j}_used')

        # pair[i][k][j] = 1 if both papers i and k are in session j
        pair = {}
        for (i, k), sim_score in self.similarity_matrix.items():
            for j in range(self.max_sessions):
                pair[i, k, j] = model.NewBoolVar(f'pair_{i}_{k}_in_session_{j}')

        print(f"  Created {len(x)} paper assignment variables")
        print(f"  Created {len(y)} session usage variables")
        print(f"  Created {len(pair)} pair tracking variables")

        # Constraints
        print("\nAdding constraints...")

        # 1. Each paper assigned to exactly one session
        for i in range(self.num_papers):
            model.Add(sum(x[i, j] for j in range(self.max_sessions)) == 1)

        # 2. Session capacity constraint (max duration)
        for j in range(self.max_sessions):
            model.Add(
                sum(self.papers[i]['minutes'] * x[i, j] for i in range(self.num_papers))
                <= self.session_duration
            )

        # 3. Session minimum fill constraint (only if session is used)
        for j in range(self.max_sessions):
            session_minutes = sum(self.papers[i]['minutes'] * x[i, j] for i in range(self.num_papers))
            # If session is used, it must meet minimum fill
            model.Add(session_minutes >= int(self.min_fill_minutes)).OnlyEnforceIf(y[j])
            # Session is used if it has any papers
            model.Add(session_minutes > 0).OnlyEnforceIf(y[j])
            model.Add(session_minutes == 0).OnlyEnforceIf(y[j].Not())

        # 4. Pair consistency constraints
        for (i, k), sim_score in self.similarity_matrix.items():
            for j in range(self.max_sessions):
                # pair[i,k,j] is true iff both x[i,j] and x[k,j] are true
                model.Add(pair[i, k, j] <= x[i, j])
                model.Add(pair[i, k, j] <= x[k, j])
                model.Add(pair[i, k, j] >= x[i, j] + x[k, j] - 1)

        print("  Constraints added successfully")

        # Objective: Maximize pairwise similarity
        print("\nSetting up objective function...")
        objective_terms = []
        for (i, k), sim_score in self.similarity_matrix.items():
            for j in range(self.max_sessions):
                objective_terms.append(sim_score * pair[i, k, j])

        model.Maximize(sum(objective_terms))
        print(f"  Objective: Maximize sum of {len(objective_terms)} weighted pair assignments")

        # Solve
        print("\nSolving with CP-SAT...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # 1 minute time limit
        solver.parameters.log_search_progress = True

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            print("\n✓ Optimal solution found!")
        elif status == cp_model.FEASIBLE:
            print("\n✓ Feasible solution found (may not be optimal)")
        else:
            print(f"\n✗ Solver status: {solver.StatusName(status)}")
            return []

        print(f"  Objective value: {solver.ObjectiveValue()}")
        print(f"  Solve time: {solver.WallTime():.2f}s")

        # Extract solution
        sessions = self._extract_solution(solver, x, y)

        return sessions

    def _extract_solution(self, solver: cp_model.CpSolver, x: Dict, y: Dict) -> List[Dict]:
        """
        Extract session assignments from solver solution.

        Args:
            solver: Solved CP-SAT solver
            x: Paper assignment variables
            y: Session usage variables

        Returns:
            List of session dictionaries
        """
        print("\nExtracting solution...")
        sessions = []

        for j in range(self.max_sessions):
            if solver.Value(y[j]) == 1:
                # This session is used
                assigned_papers = []
                total_minutes = 0

                for i in range(self.num_papers):
                    if solver.Value(x[i, j]) == 1:
                        paper = self.papers[i].copy()
                        assigned_papers.append(paper)
                        total_minutes += paper['minutes']

                if assigned_papers:
                    # Collect topics from assigned papers
                    topics = self._extract_topics(assigned_papers)

                    session = {
                        'session_id': f'FILL_{j+1:02d}',
                        'topics': topics,
                        'minutes_used': total_minutes,
                        'remaining': self.session_duration - total_minutes,
                        'total_minutes': total_minutes,
                        'unused_minutes': self.session_duration - total_minutes,
                        'utilization': total_minutes / self.session_duration,
                        'papers': [
                            {
                                'id': p['id'],
                                'title': p.get('title', ''),
                                'track': p.get('track', ''),
                                'minutes': p['minutes']
                            }
                            for p in assigned_papers
                        ]
                    }
                    sessions.append(session)
                    print(f"  Session FILL_{j+1:02d}: {len(assigned_papers)} papers, {total_minutes} min ({session['utilization']:.1%})")

        return sessions

    def _extract_topics(self, papers: List[Dict]) -> List[str]:
        """
        Extract 1-2 most common topics from papers in a session.

        Args:
            papers: Papers in the session

        Returns:
            List of 1-2 topic names
        """
        tag_counts = defaultdict(int)

        for paper in papers:
            primary = get_paper_tag(paper, 'primary_tag')
            secondary = get_paper_tag(paper, 'secondary_tag')

            if primary:
                tag_counts[primary] += 3  # Weight primary tags more
            if secondary:
                tag_counts[secondary] += 1

        # Get top 1-2 topics
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        topics = [tag for tag, count in sorted_tags[:2]]

        return topics if topics else ['Mixed Topics']
