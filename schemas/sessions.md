# Sessions JSON Format

## Description

The sessions JSON file is the output of session creation algorithms (greedy or ILP). It contains the final conference program with papers assigned to themed sessions.

## Purpose

- Final output of the session building process
- Used for program visualization and analysis
- Input to scheduling and printing tools
- Contains metrics for evaluating algorithm performance

## File Location

`data/sessions_greedy.json` or `data/sessions_ilp.json`

## Structure

The file is a JSON object with sessions, leftovers, and metadata.

## Required Fields

- `sessions` - Array of session objects
  - `session_id` - Unique identifier (e.g., "S01", "S02")
  - `topics` - Array of 1-2 topic names
  - `papers` - Array of paper objects assigned to this session
    - `id` - Paper identifier
    - `track` - Paper track
    - `minutes` - Presentation duration

## Optional Fields

- `sessions[].minutes_used` - Total minutes used by papers
- `sessions[].remaining` - Remaining capacity in minutes
- `sessions[].total_minutes` - Alias for minutes_used (compatibility)
- `sessions[].unused_minutes` - Alias for remaining (compatibility)
- `sessions[].utilization` - Utilization rate (0.0 to 1.0)
- `sessions[].papers[].title` - Paper title
- `sessions[].papers[].match` - How paper matches session (primary/secondary/tertiary/none)
- `leftovers` - Array of papers not assigned to any session
- `metrics` - Summary statistics
  - `sessions_created` - Number of sessions
  - `papers_assigned` - Number of papers assigned
  - `papers_unassigned` - Number of leftovers
  - `avg_utilization` - Average session fullness
  - `median_utilization` - Median session fullness
  - `primary_match_rate` - Fraction with primary tag match
  - `secondary_match_rate` - Fraction with secondary tag match
  - `tertiary_match_rate` - Fraction with tertiary tag match
- `algorithm` - Algorithm used (greedy, ilp, hybrid)
- `config` - Configuration used
- `generated_at` - Timestamp (ISO 8601)
- `version` - Output version

## Example

```json
{
  "sessions": [
    {
      "session_id": "S01",
      "topics": ["Software testing automation"],
      "minutes_used": 84,
      "remaining": 6,
      "total_minutes": 84,
      "unused_minutes": 6,
      "utilization": 0.933,
      "papers": [
        {
          "id": "technical_42",
          "title": "A Novel Approach to Test Generation",
          "track": "technical",
          "minutes": 12,
          "match": "primary"
        },
        {
          "id": "technical_108",
          "title": "Automated Unit Test Generation Using LLMs",
          "track": "technical",
          "minutes": 12,
          "match": "primary"
        },
        {
          "id": "demo_5",
          "title": "TestGen: An Automated Testing Tool",
          "track": "demo",
          "minutes": 7,
          "match": "primary"
        }
      ]
    },
    {
      "session_id": "S02",
      "topics": ["Program analysis", "Software security and privacy"],
      "minutes_used": 86,
      "remaining": 4,
      "utilization": 0.956,
      "papers": [
        {
          "id": "technical_15",
          "title": "Static Analysis for Vulnerability Detection",
          "track": "technical",
          "minutes": 12,
          "match": "primary"
        },
        {
          "id": "technical_73",
          "title": "Privacy Leak Detection in Mobile Apps",
          "track": "technical",
          "minutes": 12,
          "match": "secondary"
        }
      ]
    }
  ],
  "leftovers": [
    {
      "id": "nier_138",
      "title": "Exploring Novel Ideas in Testing",
      "track": "nier",
      "minutes": 7,
      "primary_tag": "Rare topic",
      "secondary_tag": "Software testing automation"
    }
  ],
  "metrics": {
    "sessions_created": 23,
    "papers_assigned": 183,
    "papers_unassigned": 13,
    "avg_utilization": 0.924,
    "median_utilization": 0.933,
    "primary_match_rate": 0.951,
    "secondary_match_rate": 0.022,
    "tertiary_match_rate": 0.027
  },
  "algorithm": "greedy",
  "generated_at": "2025-01-18T10:30:00Z",
  "version": "1.0"
}
```

## Minimal Example

```json
{
  "sessions": [
    {
      "session_id": "S01",
      "topics": ["Software testing automation"],
      "papers": [
        {
          "id": "technical_42",
          "track": "technical",
          "minutes": 12
        }
      ]
    }
  ]
}
```

## Notes

- `match` field shows how well paper fits: "primary" is best, "none" is worst
- Sessions with 2 topics are mixed sessions created to maximize paper assignments
- `total_minutes` and `unused_minutes` are compatibility fields for older tools
- Good algorithms achieve >90% utilization and >90% primary match rate
