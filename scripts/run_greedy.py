#!/usr/bin/env python3
"""
CLI entry point for the Greedy Session Builder.

Usage:
    python scripts/run_greedy.py --papers data/papers.json --output data/sessions.json

Options:
    --papers: Path to papers.json file (required)
    --session-config: Path to session_config.json (optional, for duration mapping)
    --config: Path to greedy config file (optional, uses defaults if not provided)
    --output: Path to output sessions.json file (required)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from greedy_session_builder import GreedySessionBuilder
from session_utils import (
    load_papers,
    load_config,
    load_session_config,
    enrich_papers_with_duration,
    save_sessions,
    validate_papers,
    print_session_summary,
    create_default_config
)


def main():
    """Main entry point for the greedy session builder CLI."""
    parser = argparse.ArgumentParser(
        description='Greedy Session Builder for Conference Program Creation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python scripts/run_greedy.py --papers data/papers.json --output data/sessions.json

  # With session config for duration mapping
  python scripts/run_greedy.py \\
    --papers data/papers.json \\
    --session-config data/session_config.json \\
    --output data/sessions.json

  # With custom greedy config
  python scripts/run_greedy.py \\
    --papers data/papers.json \\
    --config configs/greedy_config.json \\
    --output data/sessions.json
        """
    )

    parser.add_argument(
        '--papers',
        required=True,
        help='Path to papers.json file'
    )

    parser.add_argument(
        '--session-config',
        help='Path to session_config.json (for duration mapping by track)'
    )

    parser.add_argument(
        '--config',
        help='Path to greedy algorithm config file (optional, uses defaults if not provided)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to output sessions.json file'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate papers, do not run the algorithm'
    )

    args = parser.parse_args()

    try:
        # Load papers
        print(f"Loading papers from: {args.papers}")
        papers = load_papers(args.papers)
        print(f"  Loaded {len(papers)} papers")

        # Validate papers
        print("\nValidating papers...")
        validate_papers(papers)
        print("  Papers validated successfully")

        # If validate-only mode, exit here
        if args.validate_only:
            print("\nValidation complete. Exiting (--validate-only mode).")
            return 0

        # Determine session config path and load if needed
        session_config = None
        session_config_path = args.session_config or 'data/session_config.json'

        # Check if papers need duration enrichment
        papers_need_minutes = not all('minutes' in p for p in papers)

        # Load session config if path exists
        if Path(session_config_path).exists():
            print(f"\nLoading session config from: {session_config_path}")
            session_config = load_session_config(session_config_path)
            print("  Session config loaded")
        elif papers_need_minutes:
            print("\nWarning: Papers do not have 'minutes' field.")
            print(f"  Could not find {session_config_path}")
            print("  Please provide --session-config to map track -> minutes")
            print("  or ensure papers.json already includes 'minutes' for each paper.")
            return 1

        # Load greedy algorithm config
        # Priority: 1) --config, 2) session_config.json's session_creation_options, 3) defaults
        config = None
        if args.config:
            print(f"\nLoading algorithm config from: {args.config}")
            config = load_config(args.config)
            print("  Algorithm config loaded from --config")
        elif session_config and 'session_creation_options' in session_config:
            config = session_config['session_creation_options'].copy()
            print("\nLoading algorithm config from session_config.json")
            print("  Using session_creation_options")
        else:
            print("\nUsing default algorithm configuration")
            config = create_default_config()

        # Enrich papers with duration based on track if needed
        if session_config:
            if papers_need_minutes:
                print("\nEnriching papers with duration based on track...")
                papers = enrich_papers_with_duration(papers, session_config)
                print("  Papers enriched with duration")

            # Set session_duration from session_config if not explicitly set
            if 'session_duration' not in config:
                config['session_duration'] = session_config['sessions']['duration_minutes']
                print(f"  Using session duration from session_config: {config['session_duration']} minutes")

        # Print configuration
        print("\nConfiguration:")
        print(f"  Session duration: {config.get('session_duration')} minutes")
        print(f"  Min fill ratio: {config.get('min_fill_ratio', 0.75):.1%}")
        print(f"  Swap passes: {config.get('swap_passes', 3)}")
        print(f"  Time budget: {config.get('time_budget_seconds', 5)} seconds")
        print(f"  Weights: {config.get('weights', {})}")

        # Build sessions
        print("\n" + "="*60)
        print("BUILDING SESSIONS")
        print("="*60)

        builder = GreedySessionBuilder(papers, config)
        sessions = builder.build()

        # Export results
        print("\nExporting results...")
        sessions_dict = builder.export_sessions()
        save_sessions(sessions_dict, args.output)

        # Print summary
        print_session_summary(sessions_dict)

        print(f"✓ Successfully created {len(sessions)} sessions")
        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"\nValidation Error: {e}", file=sys.stderr)
        return 1

    except KeyError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
