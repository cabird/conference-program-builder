# Papers JSON Format

## Description

The papers JSON file contains conference papers with metadata and assigned topic tags. Each paper represents a submission that has been accepted to the conference and needs to be scheduled into a session.

## Purpose

- Input to the session creation algorithms (greedy, ILP)
- Contains tag assignments (primary, secondary, tertiary) used for clustering papers into themed sessions
- Includes presentation duration derived from paper track

## File Location

`data/papers.json`

## Structure

The file is a JSON array of paper objects.

## Required Fields

- `id` - Unique identifier (format: `track_pid`, e.g., "technical_42")
- `track` - Paper type (technical, demo, nier, jf, industry)
- `title` - Paper title

## Optional Fields

- `pid` - Original paper ID from HotCRP
- `abstract` - Paper abstract
- `authors` - Array of author objects with first/last/email/affiliation
- `topics` - Original topics from HotCRP submission
- `tags` - Assigned topic tags for session clustering
  - `primary_tag` - Most relevant topic
  - `secondary_tag` - Second most relevant topic
  - `tertiary_tag` - Third most relevant topic

## Example

```json
[
  {
    "id": "technical_42",
    "pid": 42,
    "track": "technical",
    "title": "A Novel Approach to Software Testing",
    "abstract": "This paper presents a new method for automated test generation...",
    "authors": [
      {
        "first": "Jane",
        "last": "Doe",
        "email": "jane.doe@example.edu",
        "affiliation": "Example University"
      }
    ],
    "topics": ["Software Testing", "Quality Assurance"],
    "tags": {
      "primary_tag": "Software testing automation",
      "secondary_tag": "Program analysis",
      "tertiary_tag": "Formal methods and verification"
    }
  },
  {
    "id": "demo_5",
    "pid": 5,
    "track": "demo",
    "title": "TestGen: An Automated Testing Tool",
    "tags": {
      "primary_tag": "Software testing automation"
    }
  },
  {
    "id": "nier_138",
    "pid": 138,
    "track": "nier",
    "title": "Exploring Novel Ideas in Testing",
    "authors": [
      {
        "first": "John",
        "last": "Smith",
        "affiliation": "Tech Institute"
      }
    ],
    "tags": {
      "primary_tag": "Software testing automation",
      "secondary_tag": "AI-assisted development"
    }
  }
]
```

## Notes

- Papers without tags may not be assigned to sessions
