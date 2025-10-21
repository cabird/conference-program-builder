# Session Config JSON Format

## Description

The session configuration file defines all parameters for the conference program creation pipeline, including paper durations, session settings, algorithm options, scheduling constraints, and author availability.

## Purpose

- Configure presentation durations for each paper track
- Set session capacity and time constraints
- Define greedy algorithm behavior and optimization weights
- Specify conference schedule (dates, times, rooms)
- Define author availability constraints
- Control session creation options

## File Location

`data/session_config.json`

## Structure

The file is a JSON object with multiple configuration sections.

## Required Fields

### Paper Types
- `paper_types` - Object mapping track names to presentation duration in minutes
  - Each key is a track name (e.g., "technical", "demo", "nier")
  - Each value is duration in minutes (integer)

### Sessions
- `sessions.duration_minutes` - Maximum session length in minutes (typically 90)
- `sessions.count` - Target number of sessions to create

## Optional Fields

### Session Creation Options
- `session_creation_options` - Configuration for session building algorithms
  - `algorithm` - Algorithm to use: "greedy", "clustering", etc. (default: "greedy")
  - `min_fill_ratio` - Minimum session utilization ratio (0.0-1.0, default: 0.75)
  - `allow_two_topic_sessions` - Allow sessions with 2 topics (boolean, default: false)
  - `allow_no_match_in_mixed` - Allow papers with no tag match in mixed sessions (boolean, default: false)
  - `swap_passes` - Number of local optimization passes (integer, default: 3)
  - `time_budget_seconds` - Time limit for optimization (integer, default: 30)
  - `random_seed` - Seed for reproducible results (integer, default: 42)
  - `weights` - Scoring weights for optimization
    - `utilization` - Weight for session time utilization (default: 1.0)
    - `primary` - Points for primary tag match (default: 4.0)
    - `secondary` - Points for secondary tag match (default: 2.0)
    - `tertiary` - Points for tertiary tag match (default: 0.5)

### Schedule
- `schedule` - Array of conference days with timeslots
  - Each day object contains:
    - `date` - Date in YYYY-MM-DD format
    - `day` - Day name (e.g., "Tuesday", "Wednesday")
    - `timeslots` - Array of time slots for this day
      - `time` - Time range (e.g., "10:30-12:00")
      - `duration_minutes` - Slot duration in minutes
      - `parallel_rooms` - Number of parallel sessions
      - `room_ids` - Array of room names/IDs

### Constraints
- `constraints` - Rules for scheduling and author conflicts
  - `avoid_author_conflicts` - Prevent same author in parallel sessions (boolean)
  - `author_contstraints` - Array of author availability constraints
    - Date constraint type:
      - `type` - "date_constraint"
      - `author_name` - Full author name
      - `unavailable_dates` - Array of unavailable dates (YYYY-MM-DD)
    - Timeslot constraint type:
      - `type` - "timeslot_constraint"
      - `author_name` - Full author name
      - `date` - Specific date (YYYY-MM-DD)
      - `unavailable_timeslots` - Array of time ranges (e.g., ["15:30-17:00"])

### Metadata
- `conference` - Conference name (string)
- `version` - Config version (string)

## Complete Example

```json
{
  "conference": "ASE 2025",
  "paper_types": {
    "technical": 15,
    "jf": 12,
    "demo": 8,
    "nier": 8,
    "industry": 12
  },
  "sessions": {
    "count": 26,
    "duration_minutes": 90
  },
  "session_creation_options": {
    "algorithm": "greedy",
    "min_fill_ratio": 0.75,
    "allow_two_topic_sessions": false,
    "allow_no_match_in_mixed": false,
    "swap_passes": 3,
    "time_budget_seconds": 30,
    "random_seed": 42,
    "weights": {
      "utilization": 1.0,
      "primary": 4.0,
      "secondary": 2.0,
      "tertiary": 0.5
    }
  },
  "constraints": {
    "avoid_author_conflicts": true,
    "author_contstraints": [
      {
        "type": "date_constraint",
        "author_name": "David Lo",
        "unavailable_dates": ["2023-09-13"]
      },
      {
        "type": "timeslot_constraint",
        "author_name": "Xin Xia",
        "date": "2023-09-12",
        "unavailable_timeslots": ["15:30-17:00"]
      }
    ]
  },
  "schedule": [
    {
      "date": "2023-09-12",
      "day": "Tuesday",
      "timeslots": [
        {
          "time": "10:30-12:00",
          "duration_minutes": 90,
          "parallel_rooms": 3,
          "room_ids": ["Room C", "Plenary Room 2", "Room D"]
        },
        {
          "time": "13:30-15:00",
          "duration_minutes": 90,
          "parallel_rooms": 3,
          "room_ids": ["Room C", "Plenary Room 2", "Room E"]
        },
        {
          "time": "15:30-17:00",
          "duration_minutes": 90,
          "parallel_rooms": 3,
          "room_ids": ["Room C", "Plenary Room 2", "Room D"]
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
          "parallel_rooms": 3,
          "room_ids": ["Room C", "Plenary Room 2", "Room D"]
        }
      ]
    }
  ]
}
```

## Minimal Example

```json
{
  "paper_types": {
    "technical": 15,
    "demo": 8,
    "nier": 8
  },
  "sessions": {
    "duration_minutes": 90,
    "count": 20
  },
  "session_creation_options": {
    "weights": {
      "primary": 4.0,
      "secondary": 2.0,
      "tertiary": 0.5
    }
  }
}
```

## Notes

### Weights
The `weights` section is critical as it's used throughout the pipeline:
- **Greedy session builder**: Uses weights to score paper-to-session assignments
- **Session analysis**: Uses weights to calculate cohesion scores
- Higher weights = stronger preference for that tag level when grouping papers

### Session Duration
- Typical session durations are 90 or 120 minutes
- Paper durations include Q&A time (e.g., 12 min talk + 3 min Q&A = 15 min total)

### Schedule Structure
- Schedule defines available slots for CP-SAT scheduler
- `parallel_rooms` determines maximum concurrent sessions
- Room IDs can be any string (names or numbers)
- Different days can have different room configurations

### Author Constraints
- Date constraints: Author unavailable entire day
- Timeslot constraints: Author unavailable for specific time ranges
- Used by CP-SAT scheduler to avoid conflicts
- Author names must match exactly with names in papers.json

### Fill Ratio
- `min_fill_ratio` of 0.75 means sessions must use at least 75% of available time
- Prevents very sparse sessions
- Set lower (0.6-0.7) if having trouble filling sessions

### Algorithm Options
- `allow_two_topic_sessions`: Creates mixed-topic sessions to maximize assignment rate
- `allow_no_match_in_mixed`: More aggressive - allows any papers in mixed sessions
- `swap_passes`: More passes = better optimization but slower
- `random_seed`: Use same seed for reproducible results
