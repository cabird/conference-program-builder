#!/usr/bin/env python3
"""
Generate AI-powered session titles using LLM.
Supports both Azure OpenAI and OpenAI via the llm_client module.

Reads papers and sessions, then generates academic/formal session titles
based on the papers' titles, abstracts, and tags.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from llm_client import get_title_generation_client, LLMConfigError


def load_prompt_template(prompt_file: Path) -> str:
    """Load the prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def get_paper_tag(paper: Dict, tag_type: str) -> Optional[str]:
    """Get a tag from a paper, handling both nested and root level."""
    if tag_type in paper:
        return paper[tag_type]
    if isinstance(paper.get('tags'), dict) and tag_type in paper['tags']:
        return paper['tags'][tag_type]
    return None


def format_papers_info(papers: List[Dict], paper_lookup: Dict[str, Dict]) -> str:
    """
    Format paper information for the prompt.

    Args:
        papers: List of paper references from session
        paper_lookup: Dictionary mapping paper ID to full paper data

    Returns:
        Formatted string with paper details
    """
    papers_text = []

    for i, paper_ref in enumerate(papers, 1):
        # Extract paper ID
        if isinstance(paper_ref, str):
            paper_id = paper_ref
        else:
            paper_id = paper_ref.get('id')

        paper = paper_lookup.get(paper_id)
        if not paper:
            continue

        paper_text = f"""
### Paper {i}
**Title:** {paper.get('title', 'Untitled')}
**Track:** {paper.get('track', 'unknown')}
**Abstract:** {paper.get('abstract', 'No abstract available')}
"""
        papers_text.append(paper_text.strip())

    return "\n\n".join(papers_text)


def generate_session_title(
    client,
    session: Dict,
    paper_lookup: Dict[str, Dict],
    prompt_template: str,
    max_retries: int = 3
) -> Optional[Dict[str, str]]:
    """
    Generate a session title using LLM.

    Args:
        client: LLM client (Azure OpenAI or OpenAI)
        session: Session data
        paper_lookup: Paper lookup dictionary
        prompt_template: Prompt template string
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with 'title' and 'reasoning', or None if failed
    """
    session_id = session.get('session_id', 'Unknown')

    # Format paper information
    papers_info = format_papers_info(session.get('papers', []), paper_lookup)

    # Fill in the prompt template
    prompt = prompt_template.replace('{{papers_info}}', papers_info)

    # Try up to max_retries times
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...", end='', flush=True)

            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert conference program committee member. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            # Parse the response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate the response has required fields
            if 'title' not in result:
                raise ValueError("Response missing 'title' field")

            print(" ✓")
            return result

        except json.JSONDecodeError as e:
            print(f" ✗ JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry

        except Exception as e:
            error_msg = str(e)
            print(f" ✗ Error: {error_msg}")

            # Check for rate limit error
            if 'rate' in error_msg.lower() or '429' in error_msg:
                print(f"  Rate limited. Waiting 60 seconds...")
                time.sleep(60)
            elif attempt < max_retries - 1:
                # Exponential backoff for other errors
                wait_time = 2 ** attempt
                time.sleep(wait_time)

    print(f"  Failed after {max_retries} attempts")
    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate AI-powered session titles using LLM (Azure OpenAI or OpenAI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate titles for all sessions
  python scripts/generate_session_titles.py \\
    --papers data/papers.json \\
    --sessions data/sessions.json

  # Force regenerate all titles (even if they already exist)
  python scripts/generate_session_titles.py \\
    --papers data/papers.json \\
    --sessions data/sessions.json \\
    --force
        """
    )

    parser.add_argument(
        '--papers',
        required=True,
        help='Path to papers.json file'
    )

    parser.add_argument(
        '--sessions',
        required=True,
        help='Path to sessions.json file (will be modified in-place)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate titles even if AI_generated_title already exists'
    )

    parser.add_argument(
        '--prompt',
        default='prompts/session_title_generation.txt',
        help='Path to prompt template file (default: prompts/session_title_generation.txt)'
    )

    args = parser.parse_args()

    try:
        # Get LLM client
        print("Initializing LLM client...")
        try:
            client = get_title_generation_client()
            print("  LLM client initialized successfully")
        except LLMConfigError as e:
            print(f"Error: LLM configuration error: {e}", file=sys.stderr)
            return 1

        # Load prompt template
        print(f"\nLoading prompt template from: {args.prompt}")
        prompt_template = load_prompt_template(Path(args.prompt))

        # Load papers
        print(f"\nLoading papers from: {args.papers}")
        with open(args.papers, 'r') as f:
            papers_data = json.load(f)
        paper_lookup = {p['id']: p for p in papers_data}
        print(f"  Loaded {len(papers_data)} papers")

        # Load sessions
        print(f"\nLoading sessions from: {args.sessions}")
        with open(args.sessions, 'r') as f:
            sessions_data = json.load(f)
        sessions = sessions_data.get('sessions', [])
        print(f"  Loaded {len(sessions)} sessions")

        # Process each session
        print("\n" + "="*60)
        print("GENERATING SESSION TITLES")
        print("="*60 + "\n")

        modified_count = 0
        skipped_count = 0
        failed_count = 0

        for i, session in enumerate(sessions, 1):
            session_id = session.get('session_id', f'Session {i}')

            # Check if title already exists
            if 'AI_generated_title' in session and not args.force:
                print(f"[{i}/{len(sessions)}] {session_id}: Skipping (title already exists)")
                skipped_count += 1
                continue

            print(f"\n[{i}/{len(sessions)}] {session_id}")
            print("-" * 60)

            # Display session info
            session_topics = session.get('topics', [])
            if isinstance(session_topics, str):
                session_topics = [session_topics]
            elif 'topic' in session:
                session_topics = [session['topic']]

            if session_topics:
                print(f"Session Topics: {', '.join(session_topics)}")

            # Display paper titles
            print(f"Papers ({len(session.get('papers', []))}):")
            for j, paper_ref in enumerate(session.get('papers', []), 1):
                # Extract paper ID
                if isinstance(paper_ref, str):
                    paper_id = paper_ref
                else:
                    paper_id = paper_ref.get('id')

                paper = paper_lookup.get(paper_id)
                if paper:
                    title = paper.get('title', 'Untitled')
                    track = paper.get('track', 'unknown')
                    print(f"  {j}. [{track}] {title}")

            print("\nGenerating title...")

            result = generate_session_title(
                client,
                session,
                paper_lookup,
                prompt_template
            )

            if result:
                session['AI_generated_title'] = result['title']
                if 'reasoning' in result:
                    session['AI_title_reasoning'] = result['reasoning']
                print(f"\n✓ Generated Title: \"{result['title']}\"")
                if 'reasoning' in result:
                    print(f"  Reasoning: {result['reasoning']}")
                modified_count += 1
            else:
                print(f"\n✗ Failed to generate title")
                failed_count += 1

        # Save updated sessions
        print(f"\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        print(f"\nUpdated sessions: {modified_count}")
        print(f"Skipped sessions: {skipped_count}")
        print(f"Failed sessions:  {failed_count}")

        if modified_count > 0:
            print(f"\nSaving to: {args.sessions}")
            with open(args.sessions, 'w') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
            print("✓ Sessions file updated successfully")
        else:
            print("\nNo sessions were modified")

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
