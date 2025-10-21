#!/usr/bin/env python3
"""
CLI entry point for Fill-in Session Builder.

Usage:
    python scripts/run_fill_in.py --papers data/papers.json --sessions data/sessions.json --output fill_in_sessions.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fill_in_sessions import FillInSessionBuilder, get_paper_tag
from session_utils import load_papers, load_session_config, save_sessions, enrich_papers_with_duration


def load_sessions(filepath: str) -> dict:
    """
    Load sessions from JSON file.

    Args:
        filepath: Path to sessions JSON file

    Returns:
        Sessions dictionary with 'sessions', 'leftovers', 'metrics', etc.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def identify_leftover_papers(all_papers: list, sessions_data: dict) -> list:
    """
    Identify papers not yet assigned to any session.

    Args:
        all_papers: All papers from papers.json
        sessions_data: Session data with assigned papers

    Returns:
        List of unassigned papers
    """
    # Get set of assigned paper IDs
    assigned_ids = set()
    for session in sessions_data.get('sessions', []):
        for paper in session.get('papers', []):
            assigned_ids.add(paper['id'])

    # Also check leftovers if they exist
    for leftover in sessions_data.get('leftovers', []):
        # Leftovers are still "processed" papers, not truly unassigned
        pass

    # Find papers not in assigned set
    leftover_papers = [p for p in all_papers if p['id'] not in assigned_ids]

    return leftover_papers


def main():
    """Main entry point for fill-in session builder CLI."""
    parser = argparse.ArgumentParser(
        description='Fill-in Session Builder for leftover papers using CP-SAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_fill_in.py \\
    --papers data/papers.json \\
    --sessions data/sessions.json \\
    --output fill_in_sessions.json

  # With custom session config
  python scripts/run_fill_in.py \\
    --papers data/papers.json \\
    --sessions data/sessions.json \\
    --session-config data/session_config.json \\
    --output fill_in_sessions.json
        """
    )

    parser.add_argument(
        '--papers',
        required=True,
        help='Path to papers.json file (all papers)'
    )

    parser.add_argument(
        '--sessions',
        required=True,
        help='Path to sessions.json file (already assigned sessions)'
    )

    parser.add_argument(
        '--session-config',
        help='Path to session_config.json (optional, uses data/session_config.json by default)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to output fill_in_sessions.json file'
    )

    args = parser.parse_args()

    try:
        # Load all papers
        print(f"Loading all papers from: {args.papers}")
        all_papers = load_papers(args.papers)
        print(f"  Loaded {len(all_papers)} total papers")

        # Load existing sessions
        print(f"\nLoading existing sessions from: {args.sessions}")
        sessions_data = load_sessions(args.sessions)
        existing_sessions = sessions_data.get('sessions', [])
        print(f"  Found {len(existing_sessions)} existing sessions")

        # Load session config first
        session_config_path = args.session_config or 'data/session_config.json'
        print(f"\nLoading session config from: {session_config_path}")
        session_config = load_session_config(session_config_path)

        # Enrich all papers with duration based on track
        print("\nEnriching papers with duration based on track...")
        all_papers = enrich_papers_with_duration(all_papers, session_config)
        print("  Papers enriched with duration")

        # Identify leftover papers
        print("\nIdentifying leftover papers...")
        leftover_papers = identify_leftover_papers(all_papers, sessions_data)
        print(f"  Found {len(leftover_papers)} unassigned papers")

        if not leftover_papers:
            print("\n✓ No leftover papers to assign. All papers already in sessions!")
            return 0

        # Get session creation options
        config = session_config.get('session_creation_options', {})
        config['session_duration'] = session_config['sessions']['duration_minutes']

        print(f"\nConfiguration:")
        print(f"  Session duration: {config.get('session_duration')} minutes")
        print(f"  Min fill ratio: {config.get('min_fill_ratio', 0.75):.1%}")

        # Build fill-in sessions
        print("\n" + "="*60)
        print("BUILDING FILL-IN SESSIONS")
        print("="*60)

        builder = FillInSessionBuilder(leftover_papers, config)
        fill_in_sessions = builder.build()

        # Prepare output
        output_data = {
            'sessions': fill_in_sessions,
            'metrics': {
                'sessions_created': len(fill_in_sessions),
                'papers_assigned': sum(len(s['papers']) for s in fill_in_sessions),
                'papers_unassigned': len(leftover_papers) - sum(len(s['papers']) for s in fill_in_sessions),
                'avg_utilization': sum(s['utilization'] for s in fill_in_sessions) / len(fill_in_sessions) if fill_in_sessions else 0
            },
            'algorithm': 'cp-sat-pairwise',
            'source': 'fill_in_sessions.py'
        }

        # Save output
        print(f"\nSaving fill-in sessions to: {args.output}")
        save_sessions(output_data, args.output)

        # Print summary
        print("\n" + "="*60)
        print("FILL-IN SESSION SUMMARY")
        print("="*60)
        print(f"\nSessions created: {output_data['metrics']['sessions_created']}")
        print(f"Papers assigned: {output_data['metrics']['papers_assigned']}")
        print(f"Papers unassigned: {output_data['metrics']['papers_unassigned']}")
        print(f"Average utilization: {output_data['metrics']['avg_utilization']:.1%}")

        print(f"\n✓ Successfully created {len(fill_in_sessions)} fill-in sessions")
        print(f"✓ Output saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"\nValidation Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
