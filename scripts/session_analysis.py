#!/usr/bin/env python3
"""
Analyze conference session assignments and tag alignment.
Provides detailed statistics on how papers are grouped by topic tags.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_sessions(sessions_data: Dict, papers_data: List[Dict], session_config: Dict) -> Dict[str, Any]:
    """
    Analyze session composition and tag alignment.
    Returns comprehensive statistics.

    Cohesion scoring weights are loaded from session_config:
    - Primary tag match: session_config['session_creation_options']['weights']['primary']
    - Secondary tag match: session_config['session_creation_options']['weights']['secondary']
    - Tertiary tag match: session_config['session_creation_options']['weights']['tertiary']
    - No match: 0 points

    Args:
        sessions_data: Session data with 'sessions' key
        papers_data: List of paper dictionaries
        session_config: Configuration containing weights under session_creation_options.weights

    Raises:
        ValueError: If weights are not found in session_config
    """
    # Build paper lookup
    paper_lookup = {p['id']: p for p in papers_data}

    sessions = sessions_data['sessions']

    # Load cohesion score weights from config (required)
    if not session_config:
        raise ValueError("session_config is required but was not provided")

    if 'session_creation_options' not in session_config:
        raise ValueError("session_config missing 'session_creation_options' section")

    if 'weights' not in session_config['session_creation_options']:
        raise ValueError("session_config missing 'session_creation_options.weights' section")

    weights = session_config['session_creation_options']['weights']

    required_weights = ['primary', 'secondary', 'tertiary']
    for weight_name in required_weights:
        if weight_name not in weights:
            raise ValueError(f"session_config missing weight for '{weight_name}' in session_creation_options.weights")

    COHESION_WEIGHTS = {
        'primary': weights['primary'],
        'secondary': weights['secondary'],
        'tertiary': weights['tertiary'],
        'no_match': 0
    }

    # Overall statistics
    total_papers = sum(len(s['papers']) for s in sessions)
    total_cohesion = 0  # Will calculate as we go
    total_minutes = sum(s.get('total_minutes', 0) for s in sessions)
    total_unused = sum(s.get('unused_minutes', 0) for s in sessions)

    # Tag alignment statistics
    tag_alignment_stats = {
        'primary': 0,
        'secondary': 0,
        'tertiary': 0,
        'no_match': 0
    }

    # Per-session detailed analysis
    session_details = []

    for session in sessions:
        # Handle both 'topic' (singular) and 'topics' (list)
        if 'topics' in session:
            session_topics = session['topics'] if isinstance(session['topics'], list) else [session['topics']]
        elif 'topic' in session:
            session_topics = [session['topic']]
        else:
            # Default to empty list if neither field exists
            session_topics = []

        session_id = session['session_id']

        # Count tag alignments for this session
        primary_matches = 0
        secondary_matches = 0
        tertiary_matches = 0
        no_matches = 0

        paper_details = []
        session_cohesion = 0

        for paper_info in session['papers']:
            # paper_info might be just an ID string or a dict with 'id'
            if isinstance(paper_info, str):
                paper_id = paper_info
            else:
                paper_id = paper_info.get('id') if isinstance(paper_info, dict) else paper_info

            paper = paper_lookup.get(paper_id)

            if not paper:
                continue

            tags = paper.get('tags', {})
            primary_tag = tags.get('primary_tag')
            secondary_tag = tags.get('secondary_tag')
            tertiary_tag = tags.get('tertiary_tag')

            # Determine alignment and calculate cohesion score
            # Check if any of the paper's tags match any of the session's topics
            alignment = 'no_match'
            if primary_tag and primary_tag in session_topics:
                primary_matches += 1
                tag_alignment_stats['primary'] += 1
                alignment = 'primary'
            elif secondary_tag and secondary_tag in session_topics:
                secondary_matches += 1
                tag_alignment_stats['secondary'] += 1
                alignment = 'secondary'
            elif tertiary_tag and tertiary_tag in session_topics:
                tertiary_matches += 1
                tag_alignment_stats['tertiary'] += 1
                alignment = 'tertiary'
            else:
                no_matches += 1
                tag_alignment_stats['no_match'] += 1

            # Calculate cohesion score based on alignment
            cohesion_score = COHESION_WEIGHTS[alignment]
            session_cohesion += cohesion_score

            paper_details.append({
                'id': paper_id,
                'title': paper.get('title', 'Unknown'),
                'track': paper.get('track', 'unknown'),
                'cohesion_score': cohesion_score,
                'alignment': alignment,
                'primary_tag': primary_tag,
                'secondary_tag': secondary_tag,
                'tertiary_tag': tertiary_tag
            })

        # Calculate average cohesion score for this session
        num_papers = len(session['papers'])
        avg_cohesion = session_cohesion / num_papers if num_papers > 0 else 0
        total_cohesion += session_cohesion

        # Get time information with defaults
        session_minutes = session.get('total_minutes', 0)
        session_unused = session.get('unused_minutes', 0)
        total_time = session_minutes + session_unused

        session_details.append({
            'session_id': session_id,
            'topic': session_topics[0] if len(session_topics) == 1 else None,
            'topics': session_topics if len(session_topics) != 1 else None,
            'total_papers': num_papers,
            'primary_matches': primary_matches,
            'secondary_matches': secondary_matches,
            'tertiary_matches': tertiary_matches,
            'no_matches': no_matches,
            'total_cohesion_score': session_cohesion,
            'avg_cohesion_score': avg_cohesion,
            'total_minutes': session_minutes,
            'unused_minutes': session_unused,
            'utilization': session_minutes / total_time if total_time > 0 else 0,
            'papers': paper_details
        })

    # Sort sessions by cohesion score (descending)
    session_details.sort(key=lambda s: s['total_cohesion_score'], reverse=True)

    # Track distribution statistics - only count papers in sessions
    track_distribution = Counter()
    for session in sessions:
        for paper_info in session['papers']:
            # paper_info might be just an ID string or a dict with 'id'
            if isinstance(paper_info, str):
                paper_id = paper_info
            else:
                paper_id = paper_info.get('id') if isinstance(paper_info, dict) else paper_info

            paper = paper_lookup.get(paper_id)
            if paper:
                track_distribution[paper['track']] += 1

    # Topic distribution in sessions
    topic_distribution = Counter()
    topic_count_distribution = {1: 0, 2: 0, 3: 0}  # Count sessions by number of topics

    for s in sessions:
        if 'topics' in s:
            topics = s['topics'] if isinstance(s['topics'], list) else [s['topics']]
            for topic in topics:
                topic_distribution[topic] += 1
            # Count sessions by topic count
            num_topics = len(topics)
            if num_topics <= 3:
                topic_count_distribution[num_topics] += 1
            else:
                topic_count_distribution[3] += 1  # 3+ category
        elif 'topic' in s:
            topic_distribution[s['topic']] += 1
            topic_count_distribution[1] += 1

    return {
        'summary': {
            'num_sessions': len(sessions),
            'total_papers': total_papers,
            'avg_papers_per_session': total_papers / len(sessions) if sessions else 0,
            'total_cohesion_score': total_cohesion,
            'avg_cohesion_per_session': total_cohesion / len(sessions) if sessions else 0,
            'total_minutes_used': total_minutes,
            'total_minutes_unused': total_unused,
            'avg_utilization': total_minutes / (total_minutes + total_unused) if (total_minutes + total_unused) > 0 else 0,
            'single_topic_sessions': topic_count_distribution[1],
            'two_topic_sessions': topic_count_distribution[2],
            'three_plus_topic_sessions': topic_count_distribution[3],
            'objective_value': sessions_data.get('objective_value', 0),
            'formulation': sessions_data.get('formulation', 'Unknown')
        },
        'tag_alignment': {
            'primary_matches': tag_alignment_stats['primary'],
            'secondary_matches': tag_alignment_stats['secondary'],
            'tertiary_matches': tag_alignment_stats['tertiary'],
            'no_matches': tag_alignment_stats['no_match'],
            'primary_pct': 100 * tag_alignment_stats['primary'] / total_papers if total_papers > 0 else 0,
            'secondary_pct': 100 * tag_alignment_stats['secondary'] / total_papers if total_papers > 0 else 0,
            'tertiary_pct': 100 * tag_alignment_stats['tertiary'] / total_papers if total_papers > 0 else 0,
            'no_match_pct': 100 * tag_alignment_stats['no_match'] / total_papers if total_papers > 0 else 0
        },
        'track_distribution': dict(track_distribution),
        'topic_distribution': dict(topic_distribution),
        'sessions': session_details
    }


def print_summary(analysis: Dict):
    """Print summary statistics."""
    summary = analysis['summary']
    tag_align = analysis['tag_alignment']

    print("=" * 80)
    print("SESSION ASSIGNMENT ANALYSIS")
    print("=" * 80)
    print()

    print("OVERALL SUMMARY")
    print("-" * 80)
    print(f"Number of sessions:           {summary['num_sessions']}")
    print(f"  Single-topic sessions:      {summary['single_topic_sessions']}")
    print(f"  Two-topic sessions:         {summary['two_topic_sessions']}")
    if summary['three_plus_topic_sessions'] > 0:
        print(f"  Three+ topic sessions:      {summary['three_plus_topic_sessions']}")
    print(f"Total papers assigned:        {summary['total_papers']}")
    print(f"Avg papers per session:       {summary['avg_papers_per_session']:.1f}")
    print(f"Total cohesion score:         {summary['total_cohesion_score']:.0f}")
    print(f"Avg cohesion per session:     {summary['avg_cohesion_per_session']:.1f}")
    print()

    print("TIME UTILIZATION")
    print("-" * 80)
    print(f"Total minutes used:           {summary['total_minutes_used']}")
    print(f"Total minutes unused:         {summary['total_minutes_unused']}")
    print(f"Average utilization:          {summary['avg_utilization']:.1%}")
    print()

    print("TAG ALIGNMENT")
    print("-" * 80)
    print(f"Primary tag matches:          {tag_align['primary_matches']:3d} ({tag_align['primary_pct']:5.1f}%)")
    print(f"Secondary tag matches:        {tag_align['secondary_matches']:3d} ({tag_align['secondary_pct']:5.1f}%)")
    print(f"Tertiary tag matches:         {tag_align['tertiary_matches']:3d} ({tag_align['tertiary_pct']:5.1f}%)")
    print(f"No tag match:                 {tag_align['no_matches']:3d} ({tag_align['no_match_pct']:5.1f}%)")
    print()


def print_session_details(analysis: Dict, show_papers: bool = True, show_no_match_only: bool = False):
    """Print detailed session information."""
    print("=" * 80)
    print("SESSION DETAILS")
    print("=" * 80)
    print()

    sessions = analysis['sessions']

    if show_no_match_only:
        sessions = [s for s in sessions if s['no_matches'] > 0]
        print(f"Showing {len(sessions)} sessions with papers that don't match the session topic")
        print()

    for session in sessions:
        # Display topic or topics appropriately
        if session.get('topics'):
            topic_display = ', '.join(session['topics'])
        else:
            topic_display = session.get('topic', 'No topic')

        print(f"Session {session['session_id']}: {topic_display}")
        print("-" * 80)
        print(f"Papers: {session['total_papers']} | ", end="")
        print(f"Primary: {session['primary_matches']} | ", end="")
        print(f"Secondary: {session['secondary_matches']} | ", end="")
        print(f"Tertiary: {session['tertiary_matches']} | ", end="")
        print(f"No match: {session['no_matches']}")
        print(f"Cohesion: {session['total_cohesion_score']:.0f} (avg: {session['avg_cohesion_score']:.2f})")
        print(f"Time: {session['total_minutes']}/{session['total_minutes'] + session['unused_minutes']} min ({session['utilization']:.1%} utilized)")

        if show_papers:
            print()
            print("Papers:")
            for i, paper in enumerate(session['papers'], 1):
                alignment_symbol = {
                    'primary': '●',
                    'secondary': '◐',
                    'tertiary': '○',
                    'no_match': '✗'
                }[paper['alignment']]

                print(f"  {i:2d}. [{alignment_symbol}] {paper['title']}")
                print(f"      Track: {paper['track']:10s} | Score: {paper['cohesion_score']} | ", end="")

                if paper['alignment'] == 'no_match':
                    print(f"Tags: {paper['primary_tag'][:30]:30s} / {paper['secondary_tag'][:30]:30s}")
                else:
                    print(f"Matched via {paper['alignment']} tag")

        print()


def print_topic_distribution(analysis: Dict):
    """Print distribution of topics across sessions."""
    print("=" * 80)
    print("TOPIC DISTRIBUTION")
    print("=" * 80)
    print()

    topic_dist = analysis['topic_distribution']
    for topic, count in sorted(topic_dist.items(), key=lambda x: -x[1]):
        print(f"{count:2d} sessions: {topic}")
    print()


def print_track_distribution(analysis: Dict):
    """Print distribution of paper tracks."""
    print("=" * 80)
    print("TRACK DISTRIBUTION")
    print("=" * 80)
    print()

    track_dist = analysis['track_distribution']
    for track, count in sorted(track_dist.items(), key=lambda x: -x[1]):
        print(f"{track:12s}: {count:3d} papers")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze conference session assignments and tag alignment'
    )
    parser.add_argument(
        '--sessions',
        type=Path,
        default=Path('data/sessions.json'),
        help='Sessions JSON file (default: data/sessions.json)'
    )
    parser.add_argument(
        '--papers',
        type=Path,
        default=Path('data/papers.json'),
        help='Papers JSON file (default: data/papers.json)'
    )
    parser.add_argument(
        '--session-config',
        type=Path,
        default=Path('data/session_config.json'),
        help='Session config JSON file (default: data/session_config.json)'
    )
    parser.add_argument(
        '--show-papers',
        action='store_true',
        help='Show individual paper details for each session'
    )
    parser.add_argument(
        '--no-match-only',
        action='store_true',
        help='Show only sessions with papers that don\'t match the session topic'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        help='Save analysis results to JSON file'
    )

    args = parser.parse_args()

    try:
        # Load data
        print(f"Loading sessions from {args.sessions}")
        sessions_data = load_json(args.sessions)

        print(f"Loading papers from {args.papers}")
        papers_data = load_json(args.papers)

        print(f"Loading session config from {args.session_config}")
        session_config = load_json(args.session_config)

        print()

        # Analyze
        analysis = analyze_sessions(sessions_data, papers_data, session_config)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}", file=sys.stderr)
        return 1

    # Print results
    print_summary(analysis)
    print_track_distribution(analysis)
    print_topic_distribution(analysis)
    print_session_details(analysis, show_papers=args.show_papers, show_no_match_only=args.no_match_only)

    # Save to JSON if requested
    if args.output_json:
        print(f"Saving analysis to {args.output_json}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to {args.output_json}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
