# JSON File Format Documentation

This directory contains documentation for the JSON data files used in the conference program creation pipeline.

## Available Documentation

### 1. [papers.md](papers.md)
**File**: `data/papers.json`
**Description**: Conference papers with metadata and tag assignments

Papers are the input to the session creation algorithms. Each paper has a unique ID, track type, presentation duration, and assigned topic tags (primary, secondary, tertiary) used for clustering into themed sessions.

### 2. [tags.md](tags.md)
**File**: `data/tags.json`
**Description**: Curated taxonomy of topic tags for paper classification

The tags file defines the controlled vocabulary of topics used throughout the pipeline. Each tag has a name, description, and optional aliases to help with paper tagging.

### 3. [session_config.md](session_config.md)
**File**: `data/session_config.json`
**Description**: Configuration for session parameters and paper type durations

This configuration file defines how long each paper track gets to present, how long sessions are, and constraints like max topics per session.

### 4. [sessions.md](sessions.md)
**File**: `data/sessions_greedy.json`, `data/fill_in_sessions.json`
**Description**: Generated conference sessions with assigned papers

The output of session creation algorithms (greedy, fill-in). Contains sessions with papers organized into themed topics, plus metrics and leftover papers.

### 5. [full_session_info.md](full_session_info.md)
**File**: `data/full_session_info.json`
**Description**: Complete scheduled conference program with sessions assigned to dates, times, and rooms

The final output after running the CP-SAT scheduler. Contains the complete conference schedule ready for publication with all sessions mapped to specific timeslots and rooms, author conflicts resolved, and topic diversity enforced.

## Pipeline Flow

```
1. Paper Aggregation
   Input:  HotCRP JSON exports
   Output: data/papers.json

2. Tag Generation & Assignment
   Input:  data/papers.json
   Uses:   data/tags.json
   Output: data/papers.json (with tags added)

3. Greedy Session Creation
   Input:  data/papers.json
           data/session_config.json
   Output: data/sessions_greedy.json

4. Fill-in Session Creation (for remaining papers)
   Input:  data/papers.json
           data/sessions_greedy.json
           data/session_config.json
   Output: data/fill_in_sessions.json

5. Session Title Generation
   Input:  data/papers.json
           data/sessions_greedy.json
           data/fill_in_sessions.json
   Output: Sessions with AI_generated_title fields

6. Conference Scheduling
   Input:  data/sessions_greedy.json
           data/fill_in_sessions.json
           data/papers.json
           data/session_config.json (schedule section)
   Output: data/full_session_info.json
```

## Quick Reference

| File | Purpose | Key Fields |
|------|---------|------------|
| papers.json | Papers with tags | id, track, minutes, tags.primary_tag |
| tags.json | Topic taxonomy | tags[].name, tags[].description |
| session_config.json | Configuration | paper_types, sessions, session_creation_options.weights, schedule, constraints |
| sessions_greedy.json | Sessions from greedy algorithm | sessions[].session_id, sessions[].topic/topics, sessions[].papers |
| fill_in_sessions.json | Sessions from fill-in solver | sessions[].session_id, sessions[].topics, sessions[].papers |
| full_session_info.json | Complete scheduled program | schedule[].date, schedule[].timeslots[].sessions[].room |

## Example Usage

### Reading Papers
```python
import json

with open('data/papers.json') as f:
    papers = json.load(f)

for paper in papers:
    print(f"{paper['id']}: {paper.get('title', 'No title')}")
    print(f"  Primary tag: {paper.get('tags', {}).get('primary_tag')}")
```

### Checking Session Results
```python
import json

with open('data/sessions_greedy.json') as f:
    result = json.load(f)

print(f"Sessions created: {result['metrics']['sessions_created']}")
print(f"Papers assigned: {result['metrics']['papers_assigned']}")
print(f"Assignment rate: {result['metrics']['papers_assigned'] / (result['metrics']['papers_assigned'] + result['metrics']['papers_unassigned']):.1%}")
```

## Notes

- All JSON files use UTF-8 encoding
- Dates use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)
- IDs use snake_case format (e.g., "technical_42")
- Topic names use title case (e.g., "Software testing automation")
