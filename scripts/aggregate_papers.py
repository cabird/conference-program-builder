#!/usr/bin/env python3
"""
Aggregate papers from multiple HotCRP JSON files into a single unified dataset.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_track_name(filename: str) -> str:
    """
    Extract track name from filename.
    Example: 'ase2023-technical-data.json' -> 'technical'
    """
    # Remove file extension
    name = filename.replace('.json', '')
    # Split by hyphen and get the part before '-data'
    parts = name.split('-')
    if len(parts) >= 2 and parts[-1] == 'data':
        return parts[-2]
    # Fallback: return the part before '-data' or the whole name
    return parts[-2] if len(parts) >= 2 else name


def process_paper(paper: Dict[str, Any], track: str) -> Dict[str, Any]:
    """
    Process a single paper and extract required fields.
    """
    # Extract basic fields
    pid = paper.get('pid')
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    topics = paper.get('topics', [])

    # Process authors - extract first, last, email
    authors = []
    for author in paper.get('authors', []):
        authors.append({
            'first': author.get('first', ''),
            'last': author.get('last', ''),
            'email': author.get('email', '')
        })

    # Create unique ID
    paper_id = f"{track}_{pid}"

    return {
        'id': paper_id,
        'pid': pid,
        'track': track,
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'topics': topics
    }


def aggregate_papers(input_dir: Path, output_file: Path) -> None:
    """
    Aggregate papers from all JSON files in input directory.
    """
    logger.info(f"Reading papers from {input_dir}")

    all_papers = []
    json_files = sorted(input_dir.glob('*.json'))

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return

    for json_file in json_files:
        logger.info(f"Processing {json_file.name}")

        # Extract track name from filename
        track = extract_track_name(json_file.name)
        logger.info(f"  Track: {track}")

        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        # Process each paper
        paper_count = 0
        for paper in papers:
            # Check if this is a valid paper entry (has pid and title)
            if 'pid' in paper and 'title' in paper:
                processed_paper = process_paper(paper, track)
                all_papers.append(processed_paper)
                paper_count += 1

        logger.info(f"  Processed {paper_count} papers")

    # Write output
    logger.info(f"Writing {len(all_papers)} papers to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    logger.info("Aggregation complete!")

    # Print summary statistics
    tracks = {}
    for paper in all_papers:
        track = paper['track']
        tracks[track] = tracks.get(track, 0) + 1

    logger.info("\nSummary by track:")
    for track, count in sorted(tracks.items()):
        logger.info(f"  {track}: {count} papers")
    logger.info(f"  Total: {len(all_papers)} papers")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate papers from HotCRP JSON files'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('hotcrp_json'),
        help='Input directory containing JSON files (default: hotcrp_json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/papers.json'),
        help='Output JSON file (default: data/papers.json)'
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        return

    aggregate_papers(args.input, args.output)


if __name__ == '__main__':
    main()
