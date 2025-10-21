# Conference Program Creation Pipeline

An automated pipeline for organizing conference papers into scheduled sessions using LLM-powered tagging, greedy session allocation, constraint-based optimization, and CP-SAT scheduling. This system processes papers exported from HotCRP and uses Azure OpenAI to generate tags, then assigns papers to sessions and schedules them to specific timeslots and rooms.

## Overview

This pipeline automates the organization of research papers through multiple stages:

1. **Aggregation** - Combine papers from multiple conference tracks
2. **Tag Generation** - Discover common themes across papers
3. **Tag Assignment** - Classify each paper with relevant tags
4. **Greedy Session Allocation** - Assign papers to sessions using greedy algorithm
5. **Fill-in Session Optimization** - Use constraint solver to allocate remaining papers
6. **AI Title Generation** - Generate formal session titles using LLM
7. **Conference Scheduling** - Schedule sessions to specific dates/times/rooms using CP-SAT

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- Conference data exported from HotCRP in JSON format
- Google OR-Tools for constraint optimization

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd conference_program_creation
```

2. Install required Python packages:
```bash
pip install openai python-dotenv ortools
```

3. Set up your LLM provider credentials:

The pipeline supports both Azure OpenAI and OpenAI. Configure your provider using a `.env` file:

```bash
# Copy the example file to create your .env file
cp env.example .env

# Edit .env with your credentials
# For Azure OpenAI, set LLM_PROVIDER=azure and configure Azure variables
# For OpenAI, set LLM_PROVIDER=openai and configure OpenAI variables
```

See `env.example` for detailed configuration options and comments explaining each variable.

4. Test your LLM configuration:

```bash
python scripts/llm_client.py
```

This will:
- Validate all required environment variables are set
- Test connectivity to your LLM provider
- Verify each deployment/model is accessible
- Display a summary of your configuration

## Configuration: session_config.json

The pipeline is controlled by a central configuration file `data/session_config.json` that defines session parameters, scheduling constraints, and optimization weights.

### Key Configuration Sections

**Paper types**: Duration in minutes for each conference track
```json
"paper_types": {
  "technical": 15,    // Full research papers: 15 min presentation
  "jf": 12,          // Journal-first papers: 12 min
  "demo": 8,         // Demonstrations: 8 min
  "nier": 8,         // New ideas: 8 min
  "industry": 12     // Industry track: 12 min
}
```

**Sessions**: Number and duration of sessions
```json
"sessions": {
  "count": 26,              // Total number of 90-minute sessions to create
  "duration_minutes": 90    // Standard session length
}
```

**Session creation options**: Algorithm parameters and weights
```json
"session_creation_options": {
  "algorithm": "greedy",              // Algorithm to use (greedy, clustering, etc.)
  "min_fill_ratio": 0.75,             // Minimum 75% time utilization per session
  "allow_two_topic_sessions": false,  // Allow sessions spanning 2 topics
  "allow_no_match_in_mixed": false,   // Allow papers with no tag match in mixed sessions
  "swap_passes": 3,                   // Number of optimization passes
  "time_budget_seconds": 30,          // Time limit for optimization
  "random_seed": 42,                  // For reproducible results
  "weights": {
    "utilization": 1.0,    // Weight for session time utilization
    "primary": 4.0,        // Points for primary tag match
    "secondary": 2.0,      // Points for secondary tag match
    "tertiary": 0.5        // Points for tertiary tag match
  }
}
```

The **weights** section is particularly important as it's used throughout the pipeline:
- **Greedy session builder** uses these weights to score paper-to-session assignments
- **Session analysis** uses these weights to calculate cohesion scores
- Higher weights = stronger preference for that tag level when grouping papers

**Schedule**: Conference dates, timeslots, and rooms
```json
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
      }
    ]
  }
]
```

**Constraints**: Author availability and conflict rules
```json
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
}
```

See `schemas/session_config.md` for complete documentation of all configuration options.

## Directory Structure

```
conference_program_creation/
├── hotcrp_json/                      # Source HotCRP JSON exports
│   ├── ase2023-technical-data.json
│   ├── ase2023-demo-data.json
│   └── ...
├── data/                             # Generated data files
│   ├── papers.json                  # Aggregated and enriched papers
│   ├── tags_raw.json                # LLM-generated tags (before curation)
│   ├── tags.json                    # Curated tags with descriptions
│   ├── session_config.json          # Session/schedule configuration
│   ├── sessions_greedy.json         # Sessions from greedy allocation
│   ├── fill_in_sessions.json        # Sessions from constraint solver
│   └── full_session_info.json       # Complete scheduled sessions
├── prompts/                          # LLM prompt templates
│   ├── generate_tags.txt
│   ├── aggregate_tags.txt
│   ├── assign_tags.txt
│   └── session_title_generation.txt
├── scripts/                          # Processing scripts
│   ├── aggregate_papers.py
│   ├── generate_tags.py
│   ├── assign_tags.py
│   ├── run_greedy.py                # Greedy session allocation
│   ├── fill_in_sessions.py          # Constraint-based fill-in
│   ├── generate_session_titles.py   # AI title generation
│   ├── schedule_sessions.py         # CP-SAT scheduler
│   └── session_analysis.py          # Session quality analysis
├── schemas/                          # Data format documentation
│   ├── papers.md
│   ├── session_config.md
│   └── sessions.md
├── .env                              # Azure OpenAI credentials
└── README.md                         # This file
```

## Usage

### Step 1: Aggregate Papers

**What it does:** Combines papers from multiple HotCRP JSON export files into a single unified dataset.

**How it works:** Simple data transformation - reads all JSON files from input directory, normalizes the structure, assigns unique IDs (format: `track_pid`), and writes consolidated output.

**No AI/optimization used** - Pure data preprocessing.

```bash
python scripts/aggregate_papers.py --input hotcrp_json --output data/papers.json
```

This creates `data/papers.json` with unified paper entries containing:
- Unique ID (`track_pid` format)
- Title, abstract, authors
- Original topics from HotCRP
- Track information

### Step 2: Generate Tags

**What it does:** Discovers common research themes across the paper corpus.

**How it works:** Uses **Azure OpenAI LLM** to analyze paper titles and abstracts, then generates a taxonomy of topical tags that cover the research areas represented in the conference.

**LLM strategy (two-phase approach):**
1. **Batch generation:** Sends multiple batches of papers (default: 50 papers per batch) to the LLM, asking each batch to generate tag candidates with estimated paper counts. Each LLM call independently analyzes its batch and suggests topical tags.
2. **LLM-based aggregation:** Sends all batch results to the LLM with a second prompt asking it to merge, deduplicate, and select the best tags across all batches. The LLM identifies synonyms, consolidates related tags, and produces the final unified tag set.

This two-LLM-call approach ensures tags represent themes across the entire corpus while intelligently merging similar concepts from different batches. If only one batch is needed (small corpus), aggregation is skipped.

```bash
python scripts/generate_tags.py --input data/papers.json --output data/tags_raw.json --num-tags 20
```

Options:
- `--num-tags`: Number of tags to generate (default: 20)
- `--batch-size`: Number of papers to analyze per batch (default: 50)

This creates `data/tags_raw.json` with LLM-generated tag names ranked by frequency.

### Step 3: Curate Tags

**What it does:** Human-in-the-loop refinement of the LLM-generated tag taxonomy.

**How it works:** Manual review and editing of the tag list. This step is important because:
- LLM may generate overlapping or redundant tags
- Some tags may be too broad or too narrow
- Tags need clear descriptions for consistent application

**No automation** - Requires human judgment to select the best 15-20 tags and write clear descriptions.

1. Open `data/tags_raw.json`
2. Select the best 15-20 tags (remove duplicates, overly specific tags, etc.)
3. Add descriptions for each tag
4. Save as `data/tags.json`

Example format for `data/tags.json`:

```json
{
  "tags": [
    {
      "name": "AI-assisted development",
      "description": "Use of AI or machine learning to support or automate aspects of software engineering."
    },
    {
      "name": "Software testing",
      "description": "Techniques and automation for verifying and validating software quality."
    }
  ]
}
```

### Step 4: Assign Tags to Papers

**What it does:** Classifies each paper with primary, secondary, and tertiary topical tags.

**How it works:** Uses **Azure OpenAI LLM** to read each paper's title, abstract, and the curated tag list, then assigns the 3 most relevant tags in order of relevance.

**LLM strategy:** One API call per paper. The LLM acts as a classifier, not a generator - it only selects from the predefined tag vocabulary. Uses `response_format={"type": "json_object"}` for structured output.

```bash
python scripts/assign_tags.py --input data/papers.json --tags data/tags.json --output data/papers.json
```

Options:
- `--resume`: Resume from existing output (skip already tagged papers)
- `--delay`: Delay between API calls in seconds (default: 0.5)

This enriches `data/papers.json` with tag assignments for each paper.

### Step 5: Allocate Papers to Sessions (Greedy)

**What it does:** Assigns papers to sessions, creating topically coherent groups that fit time constraints.

**How it works:** Uses a **greedy First-Fit-Decreasing (FFD) bin packing heuristic** with local search optimization.

**Greedy strategy (multi-phase algorithm):**
1. **Phase 0 - Preparation:** Build topic pools (primary/secondary/tertiary), sort papers by duration descending (FFD strategy)
2. **Phase 1 - Primary seeding:** Create sessions for high-volume topics using only *primary tag* matches. Process topics by total volume, pack greedily (largest papers first). Only keep sessions meeting minimum fill ratio (default: 75%)
3. **Phase 2 - Secondary top-up:** Fill remaining capacity in existing sessions using papers whose *secondary tag* matches the session topic
4. **Phase 2.5 - Tertiary top-up:** Fill remaining capacity using *tertiary tag* matches
5. **Phase 3 - Mixed sessions:** Build sessions from leftover papers, allowing any tag match (but avoiding no-match papers unless configured)
6. **Phase 4 - Local search:** Three optimization passes:
   - Relocation pass: Move papers to sessions with better tag matches
   - Swap pass: Exchange papers between sessions to improve cohesion
   - Leftover swap-in: Aggressively place remaining papers by swapping with weak matches
7. **Phase 4.5 - Two-topic sessions:** Create 2-topic sessions from remaining papers by finding complementary topic pairs (enabled by default)
8. **Phase 5 - Finalization:** Compute metrics and export

**Optimization objectives:**
- Maximize session utilization (fill time slots efficiently)
- Maximize topical cohesion (primary=2.0, secondary=1.0, tertiary=0.5 weights in scoring)
- Enforce minimum fill ratio (default: 75%)
- Minimize leftover papers through aggressive swap-in strategy

**No AI used** - Deterministic algorithm based on tag matching scores and bin packing.

```bash
python scripts/run_greedy.py \
  --papers data/papers.json \
  --session-config data/session_config.json \
  --output data/sessions_greedy.json
```

This creates `data/sessions_greedy.json` with initial session allocations based on tag matching and session time constraints.

### Step 6: Fill Remaining Papers (Constraint Solver)

**What it does:** Assigns papers that the greedy algorithm couldn't place (due to low-frequency topics or time constraints).

**How it works:** Uses **Google OR-Tools CP-SAT constraint solver** to optimally pack remaining papers into new sessions by maximizing pairwise similarity.

**Constraint programming approach:**
- **Decision variables:**
  - `x[i,j]`: Binary, 1 if paper i assigned to session j
  - `y[j]`: Binary, 1 if session j is used
  - `pair[i,k,j]`: Binary, 1 if papers i and k both in session j
- **Hard constraints:**
  - Each paper assigned to exactly one session
  - Session capacity: total minutes ≤ session duration (default: 90 min)
  - Minimum fill: if session used, must have ≥ 75% capacity
  - Pair consistency: `pair[i,k,j]` true iff both papers in same session
- **Objective function:** Maximize sum of (similarity_score × pair[i,k,j]) over all paper pairs
  - Primary-primary tag match: 10 points
  - Primary-secondary cross-match: 6 points
  - Secondary-secondary match: 4 points
  - Tertiary overlap: 2 points

**Optimization:** CP-SAT uses branch-and-bound search with constraint propagation, 60-second time limit. Pre-computes all pairwise similarity scores, then finds assignment that maximizes total similarity within sessions.

**Session topics:** After assignment, extracts 1-2 most common topics from papers (primary tags weighted 3×, secondary 1×).

**No AI used** - Mathematical optimization based on tag similarity matrix.

```bash
python scripts/run_fill_in.py \
  --papers data/papers.json \
  --sessions data/sessions_greedy.json \
  --session-config data/session_config.json \
  --output data/fill_in_sessions.json
```

Options:
- `--session-config`: Path to session_config.json (optional, defaults to data/session_config.json)

This creates `data/fill_in_sessions.json` with additional sessions for remaining papers.

### Step 7: Generate Session Titles

**What it does:** Creates formal, academic session titles based on the papers assigned to each session.

**How it works:** Uses **Azure OpenAI LLM** to analyze paper titles and abstracts within each session, then generates a descriptive, professional title.

**LLM strategy:** One API call per session. Provides the LLM with all paper titles and abstracts in the session, asks for a concise (3-8 word) academic title. Returns JSON with `title` and `reasoning` fields.

**Why LLM?** Session titles require understanding semantic relationships between papers and generating natural, professional language - tasks well-suited to language models.

```bash
python scripts/generate_session_titles.py \
  --papers data/papers.json \
  --sessions data/sessions_greedy.json
```

Options:
- `--force`: Regenerate titles even if they already exist
- `--prompt`: Custom prompt template file

This adds `AI_generated_title` fields to sessions.

### Step 8: Analyze Session Quality

**What it does:** Evaluates how well papers are grouped within sessions.

**How it works:** Analyzes tag alignment between each paper and its session topic, calculating cohesion scores using weights from `session_config.json`.

**Cohesion scoring (uses weights from session_config.json):**
- Primary tag match: `session_creation_options.weights.primary` points (default: 4.0)
- Secondary tag match: `session_creation_options.weights.secondary` points (default: 2.0)
- Tertiary tag match: `session_creation_options.weights.tertiary` points (default: 0.5)
- No match: 0 points

**Metrics reported:**
- Tag alignment percentages (primary/secondary/tertiary/no match)
- Per-session cohesion scores (using configured weights)
- Time utilization statistics
- Single-topic vs two-topic session counts
- Papers that don't match their session topic

**No AI/optimization** - Simple analytical scoring with configurable weights.

```bash
python scripts/session_analysis.py \
  --sessions data/sessions_greedy.json \
  --papers data/papers.json \
  --session-config data/session_config.json \
  --show-papers
```

Options:
- `--session-config`: Path to session_config.json (default: data/session_config.json)
- `--show-papers`: Display individual paper details for each session
- `--no-match-only`: Show only sessions with mismatched papers
- `--output-json`: Save analysis to JSON file

### Step 9: Schedule Sessions to Timeslots

**What it does:** Assigns sessions to specific conference dates, times, and rooms while respecting complex constraints.

**How it works:** Uses **Google OR-Tools CP-SAT constraint solver** to find optimal schedule.

**Constraint programming approach:**
- **Decision variables:** Binary variables for each (session, timeslot, room) assignment
- **Hard constraints:**
  - Each session assigned exactly once
  - No room double-booking
  - No author conflicts (authors can't be in parallel sessions)
  - Author availability (respect speaker unavailability from config)
  - Topic diversity (no overlapping topics in parallel sessions)
- **Soft objectives (via weighted penalty terms):**
  - Room consistency: Prefer same topics in same rooms (+1.0 per match)
  - Day diversity: Penalize multiple sessions of same topic on same day (-10.0 per violation)

**Optimization:** CP-SAT uses branch-and-bound search with constraint propagation, 5-minute time limit.

**No AI used** - Mathematical optimization with hard constraints and soft preferences.

```bash
python scripts/schedule_sessions.py \
  --sessions data/sessions_greedy.json data/fill_in_sessions.json \
  --papers data/papers.json \
  --output data/full_session_info.json
```

Options:
- `--session-config`: Path to session_config.json (default: data/session_config.json)
- `--allow-topic-overlap`: Allow parallel sessions with same topics

This creates `data/full_session_info.json` with complete schedule including:
- Sessions assigned to specific dates/times/rooms
- Author conflict avoidance
- Topic diversity in parallel sessions
- Author availability constraints satisfied
- Day diversity optimization

## Output Files

### papers.json

Contains all papers with enriched metadata:

```json
{
  "id": "technical_4",
  "pid": 4,
  "track": "technical",
  "title": "LeakPair: Proactive Repairing of Memory Leaks...",
  "abstract": "Modern web applications...",
  "authors": [
    {
      "first": "Arooba",
      "last": "Shahoor",
      "email": "arooba.shahoor@gmail.com"
    }
  ],
  "topics": ["Maintenance and Evolution"],
  "tags": {
    "primary_tag": "Program repair",
    "secondary_tag": "Software testing",
    "tertiary_tag": "Web development"
  }
}
```

### tags.json

Curated tag vocabulary:

```json
{
  "tags": [
    {
      "name": "AI-assisted development",
      "description": "Use of AI or machine learning to support or automate aspects of software engineering."
    }
  ]
}
```

### sessions_greedy.json

Sessions allocated using greedy algorithm:

```json
{
  "sessions": [
    {
      "session_id": "session_1",
      "topic": "Program repair",
      "papers": ["technical_4", "technical_17", "technical_23"],
      "total_minutes": 90,
      "unused_minutes": 0
    }
  ],
  "objective_value": 452.5,
  "formulation": "greedy"
}
```

### fill_in_sessions.json

Sessions for remaining papers using constraint solver:

```json
{
  "sessions": [
    {
      "session_id": "fill_1",
      "topics": ["Software testing", "Program analysis"],
      "papers": ["technical_45", "technical_67"],
      "total_minutes": 60,
      "unused_minutes": 30
    }
  ],
  "objective_value": 127.3,
  "formulation": "fill_in"
}
```

### full_session_info.json

Complete schedule with sessions assigned to timeslots and rooms:

```json
{
  "schedule": [
    {
      "date": "2025-01-15",
      "day": "Wednesday",
      "timeslots": [
        {
          "time": "09:00-10:30",
          "duration_minutes": 90,
          "sessions": [
            {
              "session_id": "session_1",
              "topic": "Program repair",
              "room": "Room A",
              "papers": [
                {
                  "id": "technical_4",
                  "title": "LeakPair: Proactive Repairing...",
                  "track": "technical",
                  "authors": [...],
                  "minutes": 30
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "generated_at": "2025-01-10T14:30:00",
  "solver_status": "optimal"
}
```

## Common Workflows

### Full Pipeline Execution

Run all steps in sequence:

```bash
# 1. Aggregate papers from HotCRP exports
python scripts/aggregate_papers.py --input hotcrp_json --output data/papers.json

# 2. Generate tag candidates
python scripts/generate_tags.py --input data/papers.json --output data/tags_raw.json --num-tags 20

# 3. Manually curate tags (edit data/tags_raw.json → data/tags.json, select best 15-20)

# 4. Assign tags to papers
python scripts/assign_tags.py --input data/papers.json --tags data/tags.json

# 5. Allocate papers to sessions (greedy)
python scripts/run_greedy.py \
  --papers data/papers.json \
  --session-config data/session_config.json \
  --output data/sessions_greedy.json

# 6. Allocate remaining papers (constraint solver)
python scripts/fill_in_sessions.py \
  --papers data/papers.json \
  --existing-sessions data/sessions_greedy.json \
  --session-config data/session_config.json \
  --output data/fill_in_sessions.json

# 7. Generate session titles
python scripts/generate_session_titles.py \
  --papers data/papers.json \
  --sessions data/sessions_greedy.json

python scripts/generate_session_titles.py \
  --papers data/papers.json \
  --sessions data/fill_in_sessions.json

# 8. Analyze session quality
python scripts/session_analysis.py \
  --sessions data/sessions_greedy.json \
  --papers data/papers.json

# 9. Schedule sessions to conference program
python scripts/schedule_sessions.py \
  --sessions data/sessions_greedy.json data/fill_in_sessions.json \
  --papers data/papers.json \
  --output data/full_session_info.json
```

### Re-tagging Papers

If you update your tag vocabulary or descriptions:

```bash
# Edit data/tags.json with new tags/descriptions
python scripts/assign_tags.py --input data/papers.json --tags data/tags.json
```

### Resuming Interrupted Tag Assignment

If tag assignment is interrupted:

```bash
python scripts/assign_tags.py --resume
```

### Regenerating Session Titles

Force regenerate all AI-generated titles:

```bash
python scripts/generate_session_titles.py \
  --papers data/papers.json \
  --sessions data/sessions_greedy.json \
  --force
```

### Analyzing Session Quality

Compare greedy vs fill-in session cohesion:

```bash
# Greedy sessions
python scripts/session_analysis.py \
  --sessions data/sessions_greedy.json \
  --papers data/papers.json \
  --output-json analysis_greedy.json

# Fill-in sessions
python scripts/session_analysis.py \
  --sessions data/fill_in_sessions.json \
  --papers data/papers.json \
  --output-json analysis_fill_in.json
```

## Customization

### Modifying Prompts

Edit prompt templates in `prompts/` to customize LLM behavior:
- `generate_tags.txt` - Tag generation logic (batch generation)
- `aggregate_tags.txt` - Tag aggregation logic (merging batch results)
- `assign_tags.txt` - Tag assignment criteria
- `session_title_generation.txt` - Session title generation style and format

Templates use `{{variable}}` placeholders that are replaced by the scripts.

### Adjusting Session Configuration

Edit `data/session_config.json` to customize:
- **Paper types**: Duration in minutes for each track (technical, industry, demo, etc.)
- **Sessions**: Number and duration of different session types
- **Schedule**: Conference dates, timeslots, and room assignments
- **Author constraints**: Specify author availability restrictions

### Adjusting Solver Parameters

**Greedy allocation:**
- Tag matching weights are configured in the greedy algorithm modules

**Fill-in sessions:**
- `--max-time`: Increase for better solutions on large problems (default: 60s)
- `--min-similarity`: Lower threshold allows more diverse papers in sessions

**CP-SAT scheduler:**
- Solver time limit: 300s (5 minutes) by default, edit in `schedule_sessions.py`
- Day diversity penalty weight: 10 by default, adjust for stronger/weaker day spreading

## Troubleshooting

### Rate Limiting

If you encounter rate limiting errors:
- Increase the `--delay` parameter in `assign_tags.py`
- Process papers in smaller batches
- Use the `--resume` flag to continue after rate limit resets
- The title generation script automatically waits 60s on rate limit errors

### Missing Environment Variables

If scripts fail with "Missing required environment variables":
- Verify `.env` file exists in the project root
- Check all required variables are set:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_DEPLOYMENT_TAG_GENERATION`
  - `AZURE_OPENAI_DEPLOYMENT_TITLE_GENERATION`
  - `AZURE_OPENAI_API_VERSION`

### JSON Parsing Errors

If LLM responses fail to parse:
- Check the prompt templates are requesting JSON output
- Review the Azure OpenAI API response format
- The scripts use `response_format={"type": "json_object"}` for structured output

### Infeasible Scheduling

If the CP-SAT scheduler reports infeasibility:
- **Author conflicts**: Too many sessions with overlapping authors
  - Solution: Reduce parallel sessions or spread papers differently
- **Topic diversity**: Too many sessions with same topics
  - Solution: Use `--allow-topic-overlap` flag
- **Author constraints**: Authors unavailable during too many timeslots
  - Solution: Review `session_config.json` constraints
- **Insufficient capacity**: Not enough room/time slots
  - Solution: Add more timeslots or rooms in `session_config.json`

### Poor Session Cohesion

If session analysis shows low cohesion scores:
- Review tag assignments for papers
- Consider re-running greedy with different configuration
- Check if fill-in sessions need higher `--min-similarity` threshold
- Papers with `no_match` alignment may need manual review

## Advanced Usage

### Processing Specific Tracks

To process only specific tracks, filter the JSON files:

```bash
python scripts/aggregate_papers.py --input hotcrp_json
# Then manually edit data/papers.json to keep only desired tracks
```

### Custom Tag Numbers

Generate different numbers of tags for different purposes:

```bash
# Fewer tags for broad categorization
python scripts/generate_tags.py --num-tags 15

# More tags for fine-grained categorization
python scripts/generate_tags.py --num-tags 30

# Default (recommended)
python scripts/generate_tags.py --num-tags 20
```

### Testing Scheduling Without Topic Diversity

If you want to relax the topic diversity constraint:

```bash
python scripts/schedule_sessions.py \
  --sessions data/sessions_greedy.json data/fill_in_sessions.json \
  --papers data/papers.json \
  --output data/full_session_info.json \
  --allow-topic-overlap
```

### Analyzing Specific Sessions

Show only problematic sessions with papers that don't match topics:

```bash
python scripts/session_analysis.py \
  --sessions data/sessions_greedy.json \
  --papers data/papers.json \
  --show-papers \
  --no-match-only
```

## Cost Considerations

Azure OpenAI API calls incur costs based on token usage:

- **Tag generation**: ~1-2 API calls for entire corpus
- **Tag assignment**: 1 API call per paper
- **Session title generation**: 1 API call per session

For a conference with 100 papers organized into 20 sessions:
- Tag generation/assignment: ~105 API calls
- Title generation: ~20 API calls
- **Total**: ~125 API calls

The constraint solvers (greedy, fill-in, CP-SAT scheduling) run locally and incur no API costs.

## Key Features

- **Hybrid approach**: Combines greedy allocation with constraint optimization for completeness
- **Smart scheduling**: CP-SAT solver handles complex constraints (author conflicts, topic diversity, availability)
- **Quality analysis**: Built-in cohesion scoring and tag alignment metrics
- **Day diversity**: Automatically spreads topics across conference days
- **Author constraints**: Respects speaker availability and conflicts
- **AI-powered titles**: LLM generates academic session titles from paper content
- **Flexible configuration**: Easily adjust session durations, tracks, schedules, and rooms

## Documentation

- `schemas/papers.md` - Paper data format
- `schemas/session_config.md` - Configuration file format
- `schemas/sessions.md` - Session output format

## License

[Add your license here]

## Support

For issues or questions, please [add contact information or issue tracker link].
