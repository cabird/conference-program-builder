#!/usr/bin/env python3
"""
Assign primary, secondary, and tertiary tags to each paper using LLM.
Supports both Azure OpenAI and OpenAI via the llm_client module.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from llm_client import get_tag_assignment_client, LLMConfigError
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompt_template(prompt_file: Path) -> str:
    """Load prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def format_tag_list(tags: List[Dict[str, str]]) -> str:
    """
    Format tag list with descriptions for the prompt.
    """
    tag_lines = []
    for tag in tags:
        name = tag['name']
        description = tag.get('description', '')
        tag_lines.append(f"- {name}: {description}")

    return '\n'.join(tag_lines)


def validate_tags(assigned_tags: Dict[str, str], valid_tag_names: set) -> tuple[bool, list]:
    """
    Validate that assigned tags are in the allowed set.
    Returns (is_valid, list_of_invalid_tags)
    """
    invalid_tags = []
    for tag_type in ['primary_tag', 'secondary_tag', 'tertiary_tag']:
        tag = assigned_tags.get(tag_type, '')
        if tag and tag not in valid_tag_names:
            invalid_tags.append(f"{tag_type}='{tag}'")

    return len(invalid_tags) == 0, invalid_tags


def assign_tags_to_paper(
    paper: Dict[str, Any],
    prompt_template: str,
    tags: List[Dict[str, str]],
    valid_tag_names: set,
    client,
    max_retries: int = 10
) -> Dict[str, str]:
    """
    Assign tags to a single paper using LLM with validation.
    Retries with emphasis if tags are not in the allowed set.
    """
    # Format base prompt
    base_prompt = prompt_template.replace('{{title}}', paper['title'])
    base_prompt = base_prompt.replace('{{abstract}}', paper['abstract'])
    base_prompt = base_prompt.replace('{{topics}}', ', '.join(paper['topics']) if paper['topics'] else 'None')
    base_prompt = base_prompt.replace('{{tag_list}}', format_tag_list(tags))

    for attempt in range(1, max_retries + 1):
        try:
            # Add emphasis after first failed attempt
            if attempt == 1:
                prompt = base_prompt
            else:
                prompt = base_prompt + f"\n\nIMPORTANT: You MUST select tags from the provided tag list above. Do not create new tags or modify tag names. The tags you return must EXACTLY match the tag names provided."

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that categorizes academic papers using predefined tags."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)

            assigned_tags = {
                'primary_tag': result.get('primary_tag', ''),
                'secondary_tag': result.get('secondary_tag', ''),
                'tertiary_tag': result.get('tertiary_tag', '')
            }

            # Validate tags
            is_valid, invalid_tags = validate_tags(assigned_tags, valid_tag_names)

            if is_valid:
                if attempt > 1:
                    logger.info(f"    Validation succeeded on attempt {attempt}")
                return assigned_tags
            else:
                logger.warning(f"    Attempt {attempt}/{max_retries}: Invalid tags returned: {', '.join(invalid_tags)}")
                if attempt == max_retries:
                    logger.error(f"    Failed to get valid tags after {max_retries} attempts. Stopping script.")
                    raise ValueError(f"Failed to get valid tags after {max_retries} attempts for paper {paper['id']}")

        except json.JSONDecodeError as e:
            logger.error(f"    Attempt {attempt}/{max_retries}: JSON parsing error: {e}")
            if attempt == max_retries:
                raise
        except Exception as e:
            logger.error(f"    Attempt {attempt}/{max_retries}: Error: {e}")
            if attempt == max_retries:
                raise

    # Should never reach here, but just in case
    raise ValueError(f"Failed to assign tags to paper {paper['id']}")


def main():
    parser = argparse.ArgumentParser(
        description='Assign tags to papers using LLM (Azure OpenAI or OpenAI)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/papers.json'),
        help='Input papers JSON file (default: data/papers.json)'
    )
    parser.add_argument(
        '--tags',
        type=Path,
        default=Path('data/tags.json'),
        help='Curated tags JSON file (default: data/tags.json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/papers.json'),
        help='Output papers JSON file (default: data/papers.json)'
    )
    parser.add_argument(
        '--prompt',
        type=Path,
        default=Path('prompts/assign_tags.txt'),
        help='Prompt template file (default: prompts/assign_tags.txt)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-tagging of all papers, even if they already have tags'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Check required files
    if not args.input.exists():
        logger.error(f"Input file does not exist: {args.input}")
        return

    if not args.tags.exists():
        logger.error(f"Tags file does not exist: {args.tags}")
        return

    if not args.prompt.exists():
        logger.error(f"Prompt template does not exist: {args.prompt}")
        return

    # Get LLM client
    try:
        client = get_tag_assignment_client()
        logger.info("LLM client initialized successfully")
    except LLMConfigError as e:
        logger.error(f"LLM configuration error: {e}")
        return

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers")

    # Load tags
    logger.info(f"Loading tags from {args.tags}")
    with open(args.tags, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)
        tags = tags_data['tags']

    logger.info(f"Loaded {len(tags)} tags")

    # Load prompt template
    logger.info(f"Loading prompt template from {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    # Create a set of valid tag names for validation
    valid_tag_names = {tag['name'] for tag in tags}
    logger.info(f"Valid tag names: {len(valid_tag_names)} tags")

    # Count how many papers already have tags
    papers_with_tags = sum(1 for p in papers if 'tags' in p and p['tags'].get('primary_tag'))
    if papers_with_tags > 0 and not args.force:
        logger.info(f"{papers_with_tags} papers already have tags. Use --force to re-tag them.")
    elif args.force:
        logger.info("--force flag set: Re-tagging all papers")

    # Assign tags to each paper
    logger.info("Assigning tags to papers...")

    for i, paper in enumerate(papers, 1):
        # Skip if already has tags (unless --force is specified)
        if not args.force and 'tags' in paper and paper['tags'].get('primary_tag'):
            logger.info(f"[{i}/{len(papers)}] Skipping {paper['id']} (already tagged)")
            continue

        logger.info(f"[{i}/{len(papers)}] Processing {paper['id']}: {paper['title'][:60]}...")

        try:
            # Assign tags with validation
            assigned_tags = assign_tags_to_paper(
                paper,
                prompt_template,
                tags,
                valid_tag_names,
                client
            )

            # Add tags to paper
            paper['tags'] = assigned_tags

            logger.info(f"  Tags: {assigned_tags['primary_tag']}, {assigned_tags['secondary_tag']}, {assigned_tags['tertiary_tag']}")

            # Write to disk after each paper
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

            # Add delay to avoid rate limiting
            if i < len(papers):
                time.sleep(args.delay)

        except ValueError as e:
            logger.error(f"FATAL ERROR: {e}")
            logger.error("Stopping script due to repeated validation failures.")
            # Write current state before exiting
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            return

    logger.info("Tag assignment complete!")

    # Print summary statistics
    tag_counts = {}
    for paper in papers:
        if 'tags' in paper:
            primary = paper['tags'].get('primary_tag')
            if primary:
                tag_counts[primary] = tag_counts.get(primary, 0) + 1

    logger.info("\nPrimary tag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {tag}: {count} papers")


if __name__ == '__main__':
    main()
