"""
Utility functions for session creation pipeline.

Provides functions for loading papers, configurations, and exporting results.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_papers(filepath: str) -> List[Dict]:
    """
    Load papers from a JSON file.

    Args:
        filepath: Path to papers.json file

    Returns:
        List of paper dictionaries

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Papers file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Ensure papers is a list
    if isinstance(papers, dict) and 'papers' in papers:
        papers = papers['papers']

    return papers


def load_config(filepath: str) -> Dict:
    """
    Load configuration from a JSON file.

    Args:
        filepath: Path to config.json file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def load_session_config(filepath: str) -> Dict:
    """
    Load session configuration (paper types and session parameters).

    Args:
        filepath: Path to session_config.json file

    Returns:
        Session configuration dictionary with paper_types and sessions

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Session config file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def enrich_papers_with_duration(papers: List[Dict], session_config: Dict) -> List[Dict]:
    """
    Add 'minutes' field to each paper based on its track type.

    Args:
        papers: List of paper dictionaries (must have 'track' field)
        session_config: Session config with 'paper_types' mapping track -> minutes

    Returns:
        List of papers with 'minutes' field added

    Raises:
        KeyError: If a paper's track is not in the paper_types configuration
    """
    paper_types = session_config.get('paper_types', {})

    for paper in papers:
        track = paper.get('track')
        if track not in paper_types:
            raise KeyError(f"Unknown track '{track}' for paper {paper.get('id')}. "
                          f"Available tracks: {list(paper_types.keys())}")

        paper['minutes'] = paper_types[track]

    return papers


def save_sessions(sessions_dict: Dict, filepath: str) -> None:
    """
    Save sessions to a JSON file.

    Args:
        sessions_dict: Dictionary with 'sessions', 'leftovers', and 'metrics' keys
        filepath: Path where to save the sessions.json file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sessions_dict, f, indent=2)

    print(f"\nSessions saved to: {filepath}")


def validate_papers(papers: List[Dict]) -> None:
    """
    Validate that papers have required fields.

    Args:
        papers: List of paper dictionaries

    Raises:
        ValueError: If papers are missing required fields
    """
    required_fields = ['id', 'track']

    for i, paper in enumerate(papers):
        # Check required fields
        for field in required_fields:
            if field not in paper:
                raise ValueError(f"Paper at index {i} missing required field: {field}")

        # Check for recommended fields (including nested tags)
        if 'title' not in paper:
            print(f"Warning: Paper {paper.get('id', i)} missing recommended field: title")

        # Check for primary_tag (can be at root or in tags.primary_tag)
        has_primary_tag = (
            'primary_tag' in paper or
            (isinstance(paper.get('tags'), dict) and 'primary_tag' in paper['tags'])
        )
        if not has_primary_tag:
            print(f"Warning: Paper {paper.get('id', i)} missing recommended field: primary_tag (should be in 'tags.primary_tag' or at root level)")


def print_session_summary(sessions_dict: Dict) -> None:
    """
    Print a summary of the sessions created.

    Args:
        sessions_dict: Dictionary with 'sessions', 'leftovers', and 'metrics' keys
    """
    sessions = sessions_dict.get('sessions', [])
    leftovers = sessions_dict.get('leftovers', [])
    metrics = sessions_dict.get('metrics', {})

    print("\n" + "="*60)
    print("SESSION CREATION SUMMARY")
    print("="*60)

    if metrics:
        print(f"\nOverall Metrics:")
        print(f"  Sessions created: {metrics.get('sessions_created', 0)}")
        print(f"  Papers assigned: {metrics.get('papers_assigned', 0)}")
        print(f"  Papers unassigned: {metrics.get('papers_unassigned', 0)}")
        print(f"  Average utilization: {metrics.get('avg_utilization', 0):.1%}")
        print(f"  Median utilization: {metrics.get('median_utilization', 0):.1%}")
        print(f"  Primary match rate: {metrics.get('primary_match_rate', 0):.1%}")
        print(f"  Secondary match rate: {metrics.get('secondary_match_rate', 0):.1%}")

    if sessions:
        print(f"\nSession Details:")
        for session in sessions[:10]:  # Show first 10
            papers_count = len(session.get('papers', []))
            util = session.get('utilization', 0)
            topics = ', '.join(session.get('topics', []))
            print(f"  {session.get('session_id')}: {topics} - {papers_count} papers, {util:.1%} full")

        if len(sessions) > 10:
            print(f"  ... and {len(sessions) - 10} more sessions")

    if leftovers:
        print(f"\nLeftover Papers ({len(leftovers)}):")
        for paper in leftovers[:5]:  # Show first 5
            print(f"  {paper.get('id')}: {paper.get('title', 'N/A')[:60]} "
                  f"({paper.get('minutes')} min, {paper.get('primary_tag')})")

        if len(leftovers) > 5:
            print(f"  ... and {len(leftovers) - 5} more papers")

    print("="*60 + "\n")


def create_default_config() -> Dict:
    """
    Create a default configuration for the greedy session builder.

    Returns:
        Default configuration dictionary
    """
    return {
        "session_duration": 90,
        "min_fill_ratio": 0.75,
        "allow_two_topic_sessions": True,  # Set to False for single-topic sessions only
        "swap_passes": 3,
        "time_budget_seconds": 5,
        "random_seed": 42,
        "weights": {
            "utilization": 1.0,
            "primary": 2.0,
            "secondary": 1.0
        }
    }
