#!/usr/bin/env python3
"""
Greedy heuristic for session assignment to test feasibility.
This attempts to assign papers to sessions using a simple greedy algorithm
to determine if a feasible solution exists before trying to optimize with ILP.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# Cohesion scoring constants
SCORE_PRIMARY = 4
SCORE_SECONDARY = 3
SCORE_TERTIARY = 1


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_paper_score_for_tag(paper: Dict[str, Any], tag: str) -> int:
    """
    Get the cohesion score for a paper if assigned to a session with the given tag.
    Returns 4 for primary match, 3 for secondary, 1 for tertiary, 0 for no match.
    """
    tags = paper.get('tags', {})

    if tags.get('primary_tag') == tag:
        return SCORE_PRIMARY
    elif tags.get('secondary_tag') == tag:
        return SCORE_SECONDARY
    elif tags.get('tertiary_tag') == tag:
        return SCORE_TERTIARY
    else:
        return 0


def paper_can_fit_in_session(paper: Dict[str, Any], session: Dict[str, Any],
                             paper_types: Dict[str, int], session_capacity: int) -> bool:
    """Check if a paper can fit in a session based on time capacity."""
    paper_duration = paper_types[paper['track']]
    return session['total_minutes'] + paper_duration <= session_capacity


def paper_matches_session_topic(paper: Dict[str, Any], session_topic: str) -> bool:
    """Check if a paper's tags match the session topic."""
    tags = paper.get('tags', {})
    return session_topic in [tags.get('primary_tag'), tags.get('secondary_tag'), tags.get('tertiary_tag')]


def greedy_assign_papers(papers: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Attempt to assign all papers to sessions using a greedy algorithm.
    Returns (success, result_dict)
    """
    paper_types = config['paper_types']
    session_capacity = config['sessions']['duration_minutes']
    max_sessions = config['sessions']['count']

    # Sort papers by duration (descending) - place largest papers first
    sorted_papers = sorted(papers, key=lambda p: paper_types[p['track']], reverse=True)

    # Track sessions
    sessions = []
    unassigned_papers = []

    print(f"Attempting greedy assignment of {len(papers)} papers to {max_sessions} sessions...")
    print(f"Session capacity: {session_capacity} minutes\n")

    for i, paper in enumerate(sorted_papers, 1):
        paper_duration = paper_types[paper['track']]
        paper_tags = [
            paper.get('tags', {}).get('primary_tag'),
            paper.get('tags', {}).get('secondary_tag'),
            paper.get('tags', {}).get('tertiary_tag')
        ]
        paper_tags = [t for t in paper_tags if t]  # Remove None values

        if not paper_tags:
            print(f"WARNING: Paper {paper['id']} has no tags!")
            unassigned_papers.append(paper)
            continue

        # Try to find an existing session that:
        # 1. Has a topic matching one of the paper's tags
        # 2. Has enough remaining capacity
        best_session = None
        best_score = -1

        for session in sessions:
            if paper_matches_session_topic(paper, session['topic']) and \
               paper_can_fit_in_session(paper, session, paper_types, session_capacity):
                # Calculate score for this assignment
                score = get_paper_score_for_tag(paper, session['topic'])
                if score > best_score:
                    best_score = score
                    best_session = session

        if best_session:
            # Assign to existing session
            best_session['papers'].append({
                'id': paper['id'],
                'track': paper['track'],
                'minutes': paper_duration,
                'title': paper['title'],
                'cohesion_score': best_score,
                'primary_tag': paper.get('tags', {}).get('primary_tag'),
                'secondary_tag': paper.get('tags', {}).get('secondary_tag'),
                'tertiary_tag': paper.get('tags', {}).get('tertiary_tag')
            })
            best_session['total_minutes'] += paper_duration
            best_session['unused_minutes'] = session_capacity - best_session['total_minutes']
            best_session['total_cohesion_score'] += best_score
            best_session['avg_cohesion_score'] = best_session['total_cohesion_score'] / len(best_session['papers'])
        else:
            # Need to create a new session
            if len(sessions) >= max_sessions:
                print(f"\nFailed at paper {i}/{len(papers)}: {paper['id']}")
                print(f"  Duration: {paper_duration} min")
                print(f"  Tags: {', '.join(paper_tags)}")
                print(f"  All {max_sessions} sessions are full or don't match tags")
                unassigned_papers.append(paper)
                continue

            # Create new session with primary tag as topic
            primary_tag = paper.get('tags', {}).get('primary_tag')
            score = SCORE_PRIMARY

            new_session = {
                'session_id': f"S{len(sessions)+1:02d}",
                'topic': primary_tag,
                'papers': [{
                    'id': paper['id'],
                    'track': paper['track'],
                    'minutes': paper_duration,
                    'title': paper['title'],
                    'cohesion_score': score,
                    'primary_tag': paper.get('tags', {}).get('primary_tag'),
                    'secondary_tag': paper.get('tags', {}).get('secondary_tag'),
                    'tertiary_tag': paper.get('tags', {}).get('tertiary_tag')
                }],
                'total_minutes': paper_duration,
                'unused_minutes': session_capacity - paper_duration,
                'total_cohesion_score': score,
                'avg_cohesion_score': score
            }
            sessions.append(new_session)

        if (i % 20 == 0):
            print(f"Progress: {i}/{len(papers)} papers assigned, {len(sessions)} sessions used")

    success = len(unassigned_papers) == 0

    print(f"\n{'='*80}")
    print(f"GREEDY ASSIGNMENT RESULT")
    print(f"{'='*80}")
    print(f"Papers assigned:       {len(papers) - len(unassigned_papers)}/{len(papers)}")
    print(f"Sessions used:         {len(sessions)}/{max_sessions}")
    print(f"Unassigned papers:     {len(unassigned_papers)}")
    print(f"Status:                {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"{'='*80}\n")

    if unassigned_papers:
        print("Unassigned papers:")
        for p in unassigned_papers:
            tags = p.get('tags', {})
            print(f"  {p['id']:20} ({paper_types[p['track']]:2} min) - Tags: {tags.get('primary_tag', 'None')}, "
                  f"{tags.get('secondary_tag', 'None')}, {tags.get('tertiary_tag', 'None')}")

    # Calculate statistics
    if sessions:
        total_cohesion = sum(s['total_cohesion_score'] for s in sessions)
        total_minutes = sum(s['total_minutes'] for s in sessions)
        avg_utilization = total_minutes / (len(sessions) * session_capacity)

        print(f"\nStatistics:")
        print(f"  Total cohesion score:  {total_cohesion:.0f}")
        print(f"  Avg cohesion/session:  {total_cohesion/len(sessions):.1f}")
        print(f"  Avg utilization:       {avg_utilization:.1%}")
        print(f"  Avg papers/session:    {(len(papers) - len(unassigned_papers))/len(sessions):.1f}")

    result = {
        'sessions': sessions,
        'objective_value': sum(s['total_cohesion_score'] for s in sessions),
        'num_sessions_used': len(sessions),
        'num_sessions_available': max_sessions,
        'papers_assigned': len(papers) - len(unassigned_papers),
        'papers_total': len(papers),
        'unassigned_papers': [p['id'] for p in unassigned_papers],
        'scoring_weights': {
            'primary': SCORE_PRIMARY,
            'secondary': SCORE_SECONDARY,
            'tertiary': SCORE_TERTIARY
        },
        'formulation': 'Greedy'
    }

    return success, result


def main():
    parser = argparse.ArgumentParser(
        description='Greedy heuristic for session assignment feasibility test'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/papers.json'),
        help='Input papers JSON file (default: data/papers.json)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('data/session_config.json'),
        help='Session configuration file (default: data/session_config.json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/sessions_greedy.json'),
        help='Output sessions JSON file (default: data/sessions_greedy.json)'
    )

    args = parser.parse_args()

    # Check required files
    if not args.input.exists():
        print(f"ERROR: Input file does not exist: {args.input}")
        return

    if not args.config.exists():
        print(f"ERROR: Config file does not exist: {args.config}")
        return

    # Load papers
    print(f"Loading papers from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers\n")

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Run greedy assignment
    success, result = greedy_assign_papers(papers, config)

    # Write output
    if success:
        print(f"\nWriting solution to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Greedy solution saved to {args.output}")
        print("\n✓ A FEASIBLE SOLUTION EXISTS")
        print("The ILP timeout is a solver performance issue, not infeasibility.")
    else:
        print("\n✗ GREEDY HEURISTIC FAILED TO FIND FEASIBLE SOLUTION")
        print("This suggests the problem may be infeasible or very difficult to solve.")


if __name__ == '__main__':
    main()
