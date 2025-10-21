# Full Session Info JSON Format

## Description

The full session info file contains the complete conference schedule with sessions assigned to specific dates, times, and rooms. This is the final output after running the CP-SAT scheduler on session files.

## Purpose

- Final scheduled program ready for publication
- Sessions mapped to specific timeslots and rooms
- Author conflicts resolved
- Topic diversity enforced across parallel sessions
- Day diversity optimization applied
- Ready for program printing and attendee distribution

## File Location

`data/full_session_info.json`

## Structure

The file is a JSON object containing a schedule array, metadata, and solver information.

## Required Fields

- `schedule` - Array of conference day objects
  - `date` - Date in YYYY-MM-DD format
  - `day` - Day name (e.g., "Tuesday", "Wednesday")
  - `timeslots` - Array of time slot objects for this day
    - `time` - Time range (e.g., "10:30-12:00")
    - `duration_minutes` - Slot duration in minutes
    - `sessions` - Array of session objects scheduled in parallel
      - `session_id` - Unique session identifier (e.g., "session_1", "fill_1")
      - `topic` - Single topic name (if single-topic session)
      - `topics` - Array of topic names (if multi-topic session)
      - `room` - Room assignment (name or ID)
      - `papers` - Array of paper objects in this session
        - `id` - Paper identifier (e.g., "technical_42")
        - `title` - Paper title
        - `track` - Paper track/type
        - `authors` - Array of author objects
          - `first` - First name
          - `last` - Last name
          - `email` - Email address (optional)
          - `affiliation` - Institution (optional)
        - `minutes` - Presentation duration

## Optional Fields

### Session Fields
- `sessions[].AI_generated_title` - LLM-generated formal session title
- `sessions[].AI_title_reasoning` - Explanation of why title was chosen
- `sessions[].total_minutes` - Total time used by papers in session
- `sessions[].unused_minutes` - Remaining unused time in session
- `sessions[].utilization` - Session fullness ratio (0.0-1.0)

### Paper Fields
- `papers[].abstract` - Paper abstract
- `papers[].topics` - Original HotCRP topics
- `papers[].tags` - Assigned tags (primary, secondary, tertiary)
- `papers[].match` - How paper matches session topic (primary/secondary/tertiary/none)

### Metadata
- `generated_at` - ISO 8601 timestamp when schedule was created
- `solver_status` - CP-SAT solver result ("optimal", "feasible", "infeasible")
- `solver_time_seconds` - Time taken by scheduler
- `objective_value` - Optimization objective achieved
- `constraints_satisfied` - List of constraint types satisfied
- `statistics` - Summary statistics
  - `total_sessions` - Number of sessions scheduled
  - `total_papers` - Number of papers in schedule
  - `parallel_sessions` - Sessions with topic conflicts (if allowed)
  - `author_conflicts_avoided` - Number of potential conflicts prevented

## Complete Example

```json
{
  "schedule": [
    {
      "date": "2023-09-12",
      "day": "Tuesday",
      "timeslots": [
        {
          "time": "10:30-12:00",
          "duration_minutes": 90,
          "sessions": [
            {
              "session_id": "session_1",
              "topic": "Software testing automation",
              "room": "Room C",
              "AI_generated_title": "Automated Test Generation and Fault Localization",
              "total_minutes": 84,
              "unused_minutes": 6,
              "utilization": 0.933,
              "papers": [
                {
                  "id": "technical_42",
                  "title": "A Novel Approach to Test Generation",
                  "track": "technical",
                  "minutes": 15,
                  "authors": [
                    {
                      "first": "Jane",
                      "last": "Doe",
                      "email": "jane.doe@example.edu",
                      "affiliation": "Example University"
                    }
                  ],
                  "abstract": "This paper presents a new method...",
                  "tags": {
                    "primary_tag": "Software testing automation",
                    "secondary_tag": "Program analysis",
                    "tertiary_tag": "AI-assisted development"
                  },
                  "match": "primary"
                },
                {
                  "id": "technical_108",
                  "title": "LLM-Based Test Generation",
                  "track": "technical",
                  "minutes": 15,
                  "authors": [
                    {
                      "first": "John",
                      "last": "Smith",
                      "affiliation": "Tech Institute"
                    }
                  ],
                  "match": "primary"
                },
                {
                  "id": "demo_5",
                  "title": "TestGen: Automated Testing Tool Demo",
                  "track": "demo",
                  "minutes": 8,
                  "authors": [
                    {
                      "first": "Alice",
                      "last": "Johnson"
                    }
                  ],
                  "match": "primary"
                }
              ]
            },
            {
              "session_id": "session_2",
              "topic": "Program analysis",
              "room": "Plenary Room 2",
              "papers": [
                {
                  "id": "technical_15",
                  "title": "Static Analysis for Vulnerabilities",
                  "track": "technical",
                  "minutes": 15,
                  "authors": [{"first": "Bob", "last": "Williams"}],
                  "match": "primary"
                }
              ]
            },
            {
              "session_id": "fill_3",
              "topics": ["AI-assisted development", "Software maintenance and evolution"],
              "room": "Room D",
              "papers": [
                {
                  "id": "technical_89",
                  "title": "AI-Powered Code Refactoring",
                  "track": "technical",
                  "minutes": 15,
                  "authors": [{"first": "Carol", "last": "Martinez"}]
                }
              ]
            }
          ]
        },
        {
          "time": "13:30-15:00",
          "duration_minutes": 90,
          "sessions": [
            {
              "session_id": "session_4",
              "topic": "Software security and privacy",
              "room": "Room C",
              "papers": []
            }
          ]
        }
      ]
    },
    {
      "date": "2023-09-13",
      "day": "Wednesday",
      "timeslots": [
        {
          "time": "10:30-12:00",
          "duration_minutes": 90,
          "sessions": []
        }
      ]
    }
  ],
  "generated_at": "2023-09-10T14:30:00Z",
  "solver_status": "optimal",
  "solver_time_seconds": 12.5,
  "objective_value": 245.0,
  "constraints_satisfied": [
    "one_assignment_per_session",
    "no_room_conflicts",
    "author_conflict_avoidance",
    "topic_diversity",
    "author_availability"
  ],
  "statistics": {
    "total_sessions": 26,
    "total_papers": 183,
    "parallel_sessions": 0,
    "author_conflicts_avoided": 47
  }
}
```

## Minimal Example

```json
{
  "schedule": [
    {
      "date": "2023-09-12",
      "day": "Tuesday",
      "timeslots": [
        {
          "time": "10:30-12:00",
          "duration_minutes": 90,
          "sessions": [
            {
              "session_id": "session_1",
              "topic": "Software testing automation",
              "room": "Room A",
              "papers": [
                {
                  "id": "technical_42",
                  "title": "Test Generation Study",
                  "track": "technical",
                  "minutes": 15,
                  "authors": [{"first": "Jane", "last": "Doe"}]
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "generated_at": "2023-09-10T14:30:00Z",
  "solver_status": "optimal"
}
```

## Notes

### Schedule Structure
- Days are ordered chronologically
- Timeslots within each day are ordered chronologically
- Sessions within a timeslot run in parallel (same time, different rooms)
- Empty sessions arrays indicate no sessions scheduled for that slot

### Topic vs Topics
- `topic`: Single string for single-topic sessions (most common)
- `topics`: Array for multi-topic sessions (created by fill-in algorithm or when enabled)

### Author Information
- Full author details included for each paper
- Enables checking for conflicts and creating printed programs
- Email and affiliation are optional

### Solver Status
- `optimal`: Best possible solution found
- `feasible`: Valid solution found, may not be optimal
- `infeasible`: No valid solution exists (constraints too strict)

### Match Field
- Shows how well paper fits session topic
- `primary`: Paper's primary tag matches session topic (best)
- `secondary`: Paper's secondary tag matches
- `tertiary`: Paper's tertiary tag matches
- `no_match` or `none`: Paper doesn't match session topic (in mixed sessions)

### Room Assignments
- Rooms are assigned to maintain topic consistency when possible
- Same topics prefer same rooms across different timeslots
- Helps attendees find their topics easily

### Constraints Satisfied
- Lists which constraint types were successfully enforced
- Useful for debugging scheduling issues
- Common constraints: author conflicts, topic diversity, room capacity

### Day Diversity
- Scheduler tries to spread same topics across different days
- Prevents all "testing" sessions on Tuesday, for example
- Controlled by penalty weights in scheduler

### Usage
This file is the final deliverable for conference organizers. It can be:
- Imported into conference management systems
- Used to generate printed programs
- Published on conference websites
- Used for room signage and attendee apps
