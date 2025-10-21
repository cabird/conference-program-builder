#!/usr/bin/env python3
"""
Generate tags from paper titles, abstracts, and existing topics using LLM.
Supports both Azure OpenAI and OpenAI via the llm_client module.
"""

import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential
from llm_client import get_tag_generation_client, LLMConfigError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_llm_call(client, messages, response_format):
    """
    Makes a chat completion call with retry logic.
    Retries up to 6 times with exponential backoff (1-60 seconds).
    """
    return client.chat.completions.create(
        messages=messages,
        response_format=response_format
    )


def load_prompt_template(prompt_file: Path) -> str:
    """Load prompt template from file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_papers_data(papers: List[Dict[str, Any]], max_papers: int = None) -> str:
    """
    Prepare papers data for the prompt.
    Format: title, abstract, and topics for each paper.
    """
    if max_papers:
        papers = papers[:max_papers]

    papers_text = []
    for i, paper in enumerate(papers, 1):
        paper_text = f"""Paper {i}:
Title: {paper['title']}
Abstract: {paper['abstract']}
Existing Topics: {', '.join(paper['topics']) if paper['topics'] else 'None'}
"""
        papers_text.append(paper_text)

    return '\n'.join(papers_text)


def generate_tags_from_batch(
    papers: List[Dict[str, Any]],
    prompt_template: str,
    num_tags: int,
    client,
    batch_num: int,
    total_papers: int
) -> List[Dict[str, Any]]:
    """
    Generate tags from a single batch of papers.
    Returns list of tag dictionaries with name, description, and estimated_count.
    """
    logger.info(f"  Batch {batch_num}: Generating {num_tags} tags from {len(papers)} papers")

    # Prepare papers data
    papers_data = prepare_papers_data(papers, max_papers=None)

    # Format prompt
    prompt = prompt_template.replace('{{num_tags}}', str(num_tags))
    prompt = prompt.replace('{{papers_data}}', papers_data)
    prompt = prompt.replace('{{total_papers}}', str(total_papers))

    logger.info(f"  Batch {batch_num}: Calling LLM API...")

    try:
        response = make_llm_call(
            client,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes academic papers and identifies research themes."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Parse response
        content = response.choices[0].message.content
        logger.info(f"  Batch {batch_num}: Received response from API")

        # Parse JSON response
        result = json.loads(content)

        # Extract tags from various possible JSON structures
        if isinstance(result, dict):
            if 'tags' in result:
                tags = result['tags']
            elif 'tag_names' in result:
                tags = result['tag_names']
            elif 'themes' in result:
                tags = result['themes']
            else:
                # If it's a dict with a single key, use that value
                tags = list(result.values())[0] if result else []
        else:
            tags = result

        # Ensure tags is a list of dicts with name, description, and estimated_count
        normalized_tags = []
        for tag in tags:
            if isinstance(tag, str):
                # If it's just a string, convert to dict
                normalized_tags.append({
                    'name': tag,
                    'description': None,
                    'estimated_count': None
                })
            elif isinstance(tag, dict):
                # Already a dict, ensure it has the right fields
                normalized_tags.append({
                    'name': tag.get('name', tag.get('tag', str(tag))),
                    'description': tag.get('description', None),
                    'estimated_count': tag.get('estimated_count', tag.get('count', None))
                })

        logger.info(f"  Batch {batch_num}: Generated {len(normalized_tags)} tags")
        return normalized_tags

    except Exception as e:
        logger.error(f"  Batch {batch_num}: Error calling LLM API: {e}")
        raise


def aggregate_batch_tags(
    batch_results: List[List[Dict[str, Any]]],
    aggregate_prompt_template: str,
    num_tags: int,
    client
) -> List[Dict[str, Any]]:
    """
    Aggregate tags from multiple batches into final tag set.
    """
    logger.info(f"\nAggregating tags from {len(batch_results)} batches...")

    # Format batch results for the prompt
    batch_text = []
    for i, batch_tags in enumerate(batch_results, 1):
        batch_text.append(f"Batch {i} Tags:")
        for tag in batch_tags:
            batch_text.append(f"  - {tag['name']}: {tag.get('description', 'No description')} (estimated count: {tag.get('estimated_count', 0)})")
        batch_text.append("")

    batch_results_text = '\n'.join(batch_text)

    # Format prompt
    prompt = aggregate_prompt_template.replace('{{num_tags}}', str(num_tags))
    prompt = prompt.replace('{{batch_results}}', batch_results_text)

    logger.info("Calling LLM API for aggregation...")

    try:
        response = make_llm_call(
            client,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes and merges research theme classifications."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Parse response
        content = response.choices[0].message.content
        logger.info("Received aggregation response from API")

        # Parse JSON response
        result = json.loads(content)

        # Extract tags
        if isinstance(result, dict) and 'tags' in result:
            tags = result['tags']
        else:
            tags = result

        # Normalize tags
        normalized_tags = []
        for tag in tags:
            if isinstance(tag, dict):
                normalized_tags.append({
                    'name': tag.get('name', str(tag)),
                    'description': tag.get('description', None),
                    'estimated_count': tag.get('estimated_count', tag.get('count', 0))
                })

        logger.info(f"Aggregated into {len(normalized_tags)} final tags")
        return normalized_tags

    except Exception as e:
        logger.error(f"Error in aggregation: {e}")
        raise


def generate_tags(
    papers: List[Dict[str, Any]],
    prompt_template: str,
    aggregate_prompt_template: str,
    num_tags: int,
    client,
    batch_size: int = 50,
    delay_between_batches: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Generate tags using LLM with multi-batch processing.
    Process papers in batches, then aggregate results.
    Returns list of tag dictionaries with name, description, and estimated_count.
    """
    logger.info(f"Generating {num_tags} tags from {len(papers)} papers (batch size: {batch_size})")

    # Split papers into batches
    num_batches = (len(papers) + batch_size - 1) // batch_size
    logger.info(f"Processing {num_batches} batch(es)...")

    batch_results = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(papers))
        batch_papers = papers[start_idx:end_idx]

        batch_tags = generate_tags_from_batch(
            batch_papers,
            prompt_template,
            num_tags,
            client,
            i + 1,
            len(papers)
        )
        batch_results.append(batch_tags)

        # Sleep between batches to avoid rate limiting (except after last batch)
        if i < num_batches - 1:
            logger.info(f"  Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)

    # If only one batch, return those tags directly
    if len(batch_results) == 1:
        logger.info("Single batch - returning tags directly")
        return batch_results[0]

    # Aggregate results from multiple batches
    final_tags = aggregate_batch_tags(
        batch_results,
        aggregate_prompt_template,
        num_tags,
        client
    )

    return final_tags


def main():
    parser = argparse.ArgumentParser(
        description='Generate tags from papers using LLM (Azure OpenAI or OpenAI)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/papers.json'),
        help='Input papers JSON file (default: data/papers.json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/tags_raw.json'),
        help='Output tags JSON file (default: data/tags_raw.json)'
    )
    parser.add_argument(
        '--prompt',
        type=Path,
        default=Path('prompts/generate_tags.txt'),
        help='Prompt template file (default: prompts/generate_tags.txt)'
    )
    parser.add_argument(
        '--aggregate-prompt',
        type=Path,
        default=Path('prompts/aggregate_tags.txt'),
        help='Aggregation prompt template file (default: prompts/aggregate_tags.txt)'
    )
    parser.add_argument(
        '--num-tags',
        type=int,
        default=20,
        help='Number of tags to generate (default: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of papers to include in prompt (default: 50)'
    )
    parser.add_argument(
        '--delay-between-batches',
        type=float,
        default=30.0,
        help='Delay in seconds between batches to avoid rate limiting (default: 30.0)'
    )

    args = parser.parse_args()

    # Check required files
    if not args.input.exists():
        logger.error(f"Input file does not exist: {args.input}")
        return

    if not args.prompt.exists():
        logger.error(f"Prompt template does not exist: {args.prompt}")
        return

    if not args.aggregate_prompt.exists():
        logger.error(f"Aggregate prompt template does not exist: {args.aggregate_prompt}")
        return

    # Get LLM client
    try:
        client = get_tag_generation_client()
        logger.info("LLM client initialized successfully")
    except LLMConfigError as e:
        logger.error(f"LLM configuration error: {e}")
        return

    # Load papers
    logger.info(f"Loading papers from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    logger.info(f"Loaded {len(papers)} papers")

    # Load prompt templates
    logger.info(f"Loading prompt template from {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    logger.info(f"Loading aggregate prompt template from {args.aggregate_prompt}")
    aggregate_prompt_template = load_prompt_template(args.aggregate_prompt)

    # Generate tags
    tags = generate_tags(
        papers,
        prompt_template,
        aggregate_prompt_template,
        args.num_tags,
        client,
        args.batch_size,
        args.delay_between_batches
    )

    # Write output
    logger.info(f"Writing {len(tags)} tags to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'tags': tags}, f, indent=2, ensure_ascii=False)

    logger.info("Tag generation complete!")

    # Print tags sorted by estimated count
    logger.info("\nGenerated tags (sorted by estimated paper count):")
    sorted_tags = sorted(tags, key=lambda x: x.get('estimated_count', 0) or 0, reverse=True)
    for tag in sorted_tags:
        count = tag.get('estimated_count')
        description = tag.get('description', 'No description')
        if count:
            logger.info(f"  {tag['name']}: ~{count} papers")
        else:
            logger.info(f"  {tag['name']}: (no estimate)")
        logger.info(f"    {description}")


if __name__ == '__main__':
    main()
