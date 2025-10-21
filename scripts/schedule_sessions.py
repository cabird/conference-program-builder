#!/usr/bin/env python3
"""
Schedule sessions to timeslots and rooms using CP-SAT constraint solver.

Reads session files and papers, assigns sessions to specific date/time/room slots
while respecting author conflicts, topic diversity, and room preferences.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from ortools.sat.python import cp_model


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_paper_authors(paper: Dict) -> Set[str]:
    """
    Extract all author names from a paper.

    Returns set of author names in format "FirstName LastName"
    """
    authors = set()
    for author in paper.get('authors', []):
        first = author.get('first', '').strip()
        last = author.get('last', '').strip()
        if first and last:
            authors.add(f"{first} {last}")
    return authors


def get_session_authors(session: Dict, paper_lookup: Dict[str, Dict]) -> Set[str]:
    """Get all unique authors across all papers in a session."""
    authors = set()
    for paper_ref in session.get('papers', []):
        # Extract paper ID
        if isinstance(paper_ref, str):
            paper_id = paper_ref
        else:
            paper_id = paper_ref.get('id')

        paper = paper_lookup.get(paper_id)
        if paper:
            authors.update(get_paper_authors(paper))

    return authors


def validate_author_constraints(
    author_constraints: List[Dict],
    papers: List[Dict],
    schedule: List[Dict]
) -> List[str]:
    """
    Validate author constraints against papers and schedule.

    Returns list of validation errors (empty if all valid).
    """
    errors = []

    # Build set of all authors
    all_authors = set()
    for paper in papers:
        all_authors.update(get_paper_authors(paper))

    # Build set of all dates and timeslots
    all_dates = set()
    all_timeslots = {}  # date -> set of timeslots

    for day_schedule in schedule:
        date = day_schedule['date']
        all_dates.add(date)
        all_timeslots[date] = set()
        for timeslot in day_schedule['timeslots']:
            all_timeslots[date].add(timeslot['time'])

    # Validate each constraint
    for i, constraint in enumerate(author_constraints):
        constraint_type = constraint.get('type')
        author_name = constraint.get('author_name')

        # Check author exists
        if not author_name:
            errors.append(f"Constraint {i}: Missing 'author_name' field")
            continue

        if author_name not in all_authors:
            errors.append(f"Constraint {i}: Author '{author_name}' not found in any papers")

        # Validate based on constraint type
        if constraint_type == 'date_constraint':
            unavailable_dates = constraint.get('unavailable_dates', [])
            for date in unavailable_dates:
                if date not in all_dates:
                    errors.append(
                        f"Constraint {i}: Date '{date}' for author '{author_name}' "
                        f"not found in schedule. Valid dates: {sorted(all_dates)}"
                    )

        elif constraint_type == 'timeslot_constraint':
            date = constraint.get('date')
            unavailable_timeslots = constraint.get('unavailable_timeslots', [])

            if not date:
                errors.append(f"Constraint {i}: Missing 'date' field for timeslot_constraint")
                continue

            if date not in all_dates:
                errors.append(
                    f"Constraint {i}: Date '{date}' for author '{author_name}' "
                    f"not found in schedule. Valid dates: {sorted(all_dates)}"
                )
            else:
                for timeslot in unavailable_timeslots:
                    if timeslot not in all_timeslots[date]:
                        errors.append(
                            f"Constraint {i}: Timeslot '{timeslot}' on {date} for author '{author_name}' "
                            f"not found. Valid timeslots for {date}: {sorted(all_timeslots[date])}"
                        )
        else:
            errors.append(f"Constraint {i}: Unknown constraint type '{constraint_type}'")

    return errors


def check_author_availability(
    session: Dict,
    paper_lookup: Dict[str, Dict],
    date: str,
    timeslot: str,
    author_constraints: List[Dict]
) -> Tuple[bool, List[str]]:
    """
    Check if all authors in a session are available for the given date/timeslot.

    Returns: (is_available, list_of_unavailable_authors)
    """
    session_authors = get_session_authors(session, paper_lookup)
    unavailable = []

    for constraint in author_constraints:
        author_name = constraint.get('author_name')
        if author_name not in session_authors:
            continue

        constraint_type = constraint.get('type')

        if constraint_type == 'date_constraint':
            if date in constraint.get('unavailable_dates', []):
                unavailable.append(f"{author_name} (unavailable {date})")

        elif constraint_type == 'timeslot_constraint':
            if (constraint.get('date') == date and
                timeslot in constraint.get('unavailable_timeslots', [])):
                unavailable.append(f"{author_name} (unavailable {date} {timeslot})")

    return (len(unavailable) == 0, unavailable)


def get_session_topics(session: Dict) -> Set[str]:
    """Extract topics from a session as a set."""
    topics = session.get('topics', [])
    if isinstance(topics, str):
        return {topics}
    elif isinstance(topics, list):
        return set(topics)
    elif 'topic' in session:
        return {session['topic']}
    return set()


class SessionScheduler:
    """
    CP-SAT based session scheduler with author conflicts and topic diversity.
    """

    def __init__(
        self,
        sessions: List[Dict],
        papers: List[Dict],
        schedule_structure: List[Dict],
        author_constraints: List[Dict],
        allow_topic_overlap: bool = False
    ):
        """
        Initialize the scheduler.

        Args:
            sessions: List of session dictionaries
            papers: List of paper dictionaries
            schedule_structure: Schedule structure from session_config.json
            author_constraints: Author availability constraints
            allow_topic_overlap: If True, allow sessions with same topics in parallel
        """
        self.sessions = sessions
        self.paper_lookup = {p['id']: p for p in papers}
        self.schedule_structure = schedule_structure
        self.author_constraints = author_constraints
        self.allow_topic_overlap = allow_topic_overlap

        # Build schedule index
        self.timeslots = []  # List of (date, day, time, rooms) tuples
        self.slot_index = {}  # Maps (date, time) -> slot_idx

        idx = 0
        for day_schedule in schedule_structure:
            date = day_schedule['date']
            day_name = day_schedule['day']
            for timeslot in day_schedule['timeslots']:
                time = timeslot['time']
                rooms = timeslot['room_ids']
                self.timeslots.append((date, day_name, time, rooms))
                self.slot_index[(date, time)] = idx
                idx += 1

        total_capacity = sum(len(rooms) for _, _, _, rooms in self.timeslots)

        print(f"Scheduler initialized:")
        print(f"  Sessions to schedule: {len(self.sessions)}")
        print(f"  Timeslots: {len(self.timeslots)}")
        print(f"  Total capacity: {total_capacity}")

        # Check if there's enough capacity
        if len(self.sessions) > total_capacity:
            raise ValueError(
                f"Insufficient capacity: {len(self.sessions)} sessions to schedule "
                f"but only {total_capacity} available slots. "
                f"Need {len(self.sessions) - total_capacity} more slots."
            )

    def schedule(self) -> Optional[Dict]:
        """
        Schedule sessions using CP-SAT solver.

        Returns:
            Dictionary with scheduled sessions organized by day/timeslot, or None if infeasible
        """
        print("\n" + "="*60)
        print("SCHEDULING SESSIONS WITH CP-SAT")
        print("="*60)

        model = cp_model.CpModel()

        # Decision variables: assign[session_idx][slot_idx][room_idx]
        # = 1 if session is assigned to that slot and room
        assign = {}

        print("\nCreating decision variables...")
        for i, session in enumerate(self.sessions):
            for j, (date, day, time, rooms) in enumerate(self.timeslots):
                for k, room in enumerate(rooms):
                    assign[i, j, k] = model.NewBoolVar(f'session_{i}_slot_{j}_room_{k}')

        print(f"  Created {len(assign)} assignment variables")

        # Constraint 1: Each session assigned to exactly one slot+room
        print("\nAdding constraints...")
        print("  [1/5] Each session assigned exactly once...")
        for i in range(len(self.sessions)):
            model.Add(
                sum(assign[i, j, k]
                    for j in range(len(self.timeslots))
                    for k in range(len(self.timeslots[j][3])))
                == 1
            )

        # Constraint 2: Each room/slot can have at most one session
        print("  [2/5] No double-booking of rooms...")
        for j in range(len(self.timeslots)):
            for k in range(len(self.timeslots[j][3])):
                model.Add(
                    sum(assign[i, j, k] for i in range(len(self.sessions)))
                    <= 1
                )

        # Constraint 3: Author conflicts - no same author in parallel sessions
        print("  [3/5] Author conflict avoidance...")
        author_conflict_count = 0
        for j in range(len(self.timeslots)):
            # For each timeslot, check all pairs of sessions
            for i1 in range(len(self.sessions)):
                authors1 = get_session_authors(self.sessions[i1], self.paper_lookup)
                if not authors1:
                    continue

                for i2 in range(i1 + 1, len(self.sessions)):
                    authors2 = get_session_authors(self.sessions[i2], self.paper_lookup)

                    # Check if there's author overlap
                    if authors1 & authors2:
                        # These sessions cannot be in parallel (same timeslot, any rooms)
                        for k1 in range(len(self.timeslots[j][3])):
                            for k2 in range(len(self.timeslots[j][3])):
                                if k1 != k2:
                                    model.Add(assign[i1, j, k1] + assign[i2, j, k2] <= 1)
                                    author_conflict_count += 1

        print(f"    Added {author_conflict_count} author conflict constraints")

        # Constraint 4: Author availability constraints
        print("  [4/5] Author availability constraints...")
        availability_constraint_count = 0
        for i, session in enumerate(self.sessions):
            for j, (date, day, time, rooms) in enumerate(self.timeslots):
                is_available, unavailable = check_author_availability(
                    session, self.paper_lookup, date, time, self.author_constraints
                )

                if not is_available:
                    # This session cannot be scheduled in this timeslot
                    for k in range(len(rooms)):
                        model.Add(assign[i, j, k] == 0)
                        availability_constraint_count += 1

        print(f"    Added {availability_constraint_count} availability constraints")

        # Constraint 5: Topic diversity - no overlapping topics in parallel sessions
        if not self.allow_topic_overlap:
            print("  [5/5] Topic diversity (no overlap in parallel sessions)...")
            topic_constraint_count = 0
            for j in range(len(self.timeslots)):
                # For each timeslot, check all pairs of sessions
                for i1 in range(len(self.sessions)):
                    topics1 = get_session_topics(self.sessions[i1])
                    if not topics1:
                        continue

                    for i2 in range(i1 + 1, len(self.sessions)):
                        topics2 = get_session_topics(self.sessions[i2])

                        # Check if there's topic overlap
                        if topics1 & topics2:
                            # These sessions cannot be in parallel
                            for k1 in range(len(self.timeslots[j][3])):
                                for k2 in range(len(self.timeslots[j][3])):
                                    if k1 != k2:
                                        model.Add(assign[i1, j, k1] + assign[i2, j, k2] <= 1)
                                        topic_constraint_count += 1

            print(f"    Added {topic_constraint_count} topic diversity constraints")
        else:
            print("  [5/5] Topic diversity (skipped - overlap allowed)")

        # Objective: Balance room consistency and day diversity
        print("\nSetting up objective function...")
        objective_terms = []

        # Group sessions by their primary topic
        topic_sessions = defaultdict(list)
        for i, session in enumerate(self.sessions):
            topics = get_session_topics(session)
            if topics:
                primary_topic = sorted(topics)[0]  # Use first topic alphabetically
                topic_sessions[primary_topic].append(i)

        # Part 1: Prefer same topics in same rooms (positive contribution)
        room_consistency_terms = []
        for topic, session_indices in topic_sessions.items():
            if len(session_indices) > 1:
                # Create variables for room usage
                for room_idx in range(max(len(rooms) for _, _, _, rooms in self.timeslots)):
                    # Count how many of these sessions use this room
                    room_usage = []
                    for sess_idx in session_indices:
                        for slot_idx in range(len(self.timeslots)):
                            if room_idx < len(self.timeslots[slot_idx][3]):
                                room_usage.append(assign[sess_idx, slot_idx, room_idx])

                    if room_usage:
                        # Maximize room usage for this topic (encourages concentration)
                        room_consistency_terms.extend(room_usage)

        # Part 2: Penalize multiple sessions of same topic on same day (negative contribution)
        # Extract unique dates from schedule
        dates = []
        for date, day, time, rooms in self.timeslots:
            if date not in dates:
                dates.append(date)

        # For each topic, penalize multiple sessions on the same day
        day_diversity_penalty = []
        for topic, session_indices in topic_sessions.items():
            if len(session_indices) > 1:
                # For each date, create a binary variable indicating if 2+ sessions on this day
                for date_idx, date in enumerate(dates):
                    # Get slot indices for this date
                    slots_this_day = [j for j, (d, _, _, _) in enumerate(self.timeslots) if d == date]

                    # For each pair of sessions that could be on same day, penalize
                    for idx1 in range(len(session_indices)):
                        for idx2 in range(idx1 + 1, len(session_indices)):
                            sess1 = session_indices[idx1]
                            sess2 = session_indices[idx2]

                            # Create auxiliary variable for whether both are on this date
                            both_on_day = model.NewBoolVar(
                                f'topic_{topic}_sess_{sess1}_{sess2}_day_{date_idx}_both'
                            )

                            # Collect assignment variables for both sessions on this day
                            sess1_on_day = []
                            sess2_on_day = []
                            for slot_idx in slots_this_day:
                                for room_idx in range(len(self.timeslots[slot_idx][3])):
                                    sess1_on_day.append(assign[sess1, slot_idx, room_idx])
                                    sess2_on_day.append(assign[sess2, slot_idx, room_idx])

                            if sess1_on_day and sess2_on_day:
                                # both_on_day >= sess1_on_day[i] + sess2_on_day[j] - 1
                                # This is true when both sessions are on this day
                                for s1_var in sess1_on_day:
                                    for s2_var in sess2_on_day:
                                        model.Add(both_on_day >= s1_var + s2_var - 1)

                                # Penalize this situation (subtract from objective)
                                day_diversity_penalty.append(both_on_day)

        # Combine objectives: maximize room consistency, minimize same-day clustering
        # Weight the day diversity penalty higher to prioritize spreading across days
        penalty_weight = 10  # Higher weight = stronger preference for day diversity

        if room_consistency_terms or day_diversity_penalty:
            total_objective = sum(room_consistency_terms) - penalty_weight * sum(day_diversity_penalty)
            model.Maximize(total_objective)
            print(f"  Objective: Maximize room consistency ({len(room_consistency_terms)} terms)")
            print(f"             Minimize same-day clustering ({len(day_diversity_penalty)} penalty terms, weight={penalty_weight})")

        # Solve
        print("\nSolving with CP-SAT...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0  # 5 minute time limit
        solver.parameters.log_search_progress = True

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            print("\n✓ Optimal solution found!")
        elif status == cp_model.FEASIBLE:
            print("\n✓ Feasible solution found (may not be optimal)")
        else:
            print(f"\n✗ Solver status: {solver.StatusName(status)}")
            self._report_infeasibility(model, solver)
            return None

        print(f"  Solve time: {solver.WallTime():.2f}s")

        # Extract solution
        return self._extract_solution(solver, assign)

    def _report_infeasibility(self, model: cp_model.CpModel, solver: cp_model.CpSolver):
        """Report which constraints are likely causing infeasibility."""
        print("\n" + "="*60)
        print("INFEASIBILITY ANALYSIS")
        print("="*60)
        print("\nThe scheduling problem has no solution. Possible issues:")
        print("\n1. Author conflicts: Too many sessions with overlapping authors")
        print("2. Author constraints: Some authors unavailable during too many slots")
        print("3. Topic diversity: Too many sessions with overlapping topics")
        print("4. Insufficient capacity: Not enough room/time slots")
        print("\nSuggestions:")
        print("  - Use --allow-topic-overlap to relax topic diversity constraint")
        print("  - Review author constraints in session_config.json")
        print("  - Consider adding more timeslots or rooms")

    def _extract_solution(self, solver: cp_model.CpSolver, assign: Dict) -> Dict:
        """Extract the scheduled solution from solver."""
        print("\nExtracting solution...")

        # Organize by day -> timeslot -> sessions
        schedule_output = []

        for day_schedule in self.schedule_structure:
            date = day_schedule['date']
            day_name = day_schedule['day']

            day_output = {
                'date': date,
                'day': day_name,
                'timeslots': []
            }

            for timeslot_info in day_schedule['timeslots']:
                time = timeslot_info['time']
                slot_idx = self.slot_index[(date, time)]
                rooms = self.timeslots[slot_idx][3]

                timeslot_output = {
                    'time': time,
                    'duration_minutes': timeslot_info['duration_minutes'],
                    'sessions': []
                }

                # Find sessions assigned to this timeslot
                for i, session in enumerate(self.sessions):
                    for k, room in enumerate(rooms):
                        if solver.Value(assign[i, slot_idx, k]) == 1:
                            # This session is assigned here
                            session_copy = session.copy()
                            session_copy['room'] = room
                            timeslot_output['sessions'].append(session_copy)

                day_output['timeslots'].append(timeslot_output)

            schedule_output.append(day_output)

        return {
            'schedule': schedule_output,
            'generated_at': datetime.now().isoformat(),
            'solver_status': 'optimal' if solver.ObjectiveValue() else 'feasible'
        }


def enrich_session_with_paper_details(session: Dict, paper_lookup: Dict[str, Dict]) -> Dict:
    """
    Enrich session with full paper details (title, authors, track, etc).
    Removes abstract and tags as requested.
    """
    enriched_papers = []

    for paper_ref in session.get('papers', []):
        # Extract paper ID
        if isinstance(paper_ref, str):
            paper_id = paper_ref
        else:
            paper_id = paper_ref.get('id')

        paper = paper_lookup.get(paper_id)
        if paper:
            enriched_paper = {
                'id': paper['id'],
                'title': paper.get('title', 'Untitled'),
                'track': paper.get('track', 'unknown'),
                'authors': paper.get('authors', []),
                'minutes': paper.get('minutes', 0)
            }
            enriched_papers.append(enriched_paper)

    session_copy = session.copy()
    session_copy['papers'] = enriched_papers
    return session_copy


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Schedule sessions to timeslots and rooms using CP-SAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Schedule sessions from greedy output
  python scripts/schedule_sessions.py \\
    --sessions data/sessions_greedy.json \\
    --papers data/papers.json \\
    --output full_session_info.json

  # Schedule combined greedy + fill-in sessions
  python scripts/schedule_sessions.py \\
    --sessions data/sessions_greedy.json data/fill_in_sessions.json \\
    --papers data/papers.json \\
    --output full_session_info.json

  # Allow topic overlap if infeasible
  python scripts/schedule_sessions.py \\
    --sessions data/sessions_greedy.json \\
    --papers data/papers.json \\
    --output full_session_info.json \\
    --allow-topic-overlap
        """
    )

    parser.add_argument(
        '--sessions',
        nargs='+',
        required=True,
        help='One or more session JSON files'
    )

    parser.add_argument(
        '--papers',
        required=True,
        help='Path to papers.json file'
    )

    parser.add_argument(
        '--session-config',
        default='data/session_config.json',
        help='Path to session_config.json (default: data/session_config.json)'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to output full_session_info.json file'
    )

    parser.add_argument(
        '--allow-topic-overlap',
        action='store_true',
        help='Allow sessions with same topics in parallel timeslots'
    )

    args = parser.parse_args()

    try:
        # Load papers
        print(f"Loading papers from: {args.papers}")
        papers = load_json(args.papers)
        print(f"  Loaded {len(papers)} papers")

        # Load and merge session files
        print(f"\nLoading session files...")
        all_sessions = []
        for session_file in args.sessions:
            print(f"  Loading: {session_file}")
            session_data = load_json(session_file)
            sessions = session_data.get('sessions', [])
            all_sessions.extend(sessions)
            print(f"    Added {len(sessions)} sessions")

        print(f"\nTotal sessions to schedule: {len(all_sessions)}")

        # Load session config
        print(f"\nLoading session config from: {args.session_config}")
        session_config = load_json(args.session_config)
        schedule = session_config.get('schedule', [])
        author_constraints = session_config.get('constraints', {}).get('author_contstraints', [])

        print(f"  Schedule days: {len(schedule)}")
        print(f"  Author constraints: {len(author_constraints)}")

        # Validate author constraints
        if author_constraints:
            print("\nValidating author constraints...")
            validation_errors = validate_author_constraints(author_constraints, papers, schedule)
            if validation_errors:
                print("✗ Author constraint validation failed:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            print("  ✓ All author constraints are valid")

        # Create scheduler and solve
        scheduler = SessionScheduler(
            all_sessions,
            papers,
            schedule,
            author_constraints,
            args.allow_topic_overlap
        )

        result = scheduler.schedule()

        if result is None:
            print("\n✗ Scheduling failed (infeasible)")
            return 1

        # Enrich sessions with full paper details
        print("\nEnriching sessions with paper details...")
        paper_lookup = {p['id']: p for p in papers}

        for day in result['schedule']:
            for timeslot in day['timeslots']:
                enriched_sessions = []
                for session in timeslot['sessions']:
                    enriched = enrich_session_with_paper_details(session, paper_lookup)
                    enriched_sessions.append(enriched)
                timeslot['sessions'] = enriched_sessions

        # Save output
        print(f"\nSaving complete schedule to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "="*60)
        print("SCHEDULING SUMMARY")
        print("="*60)

        for day in result['schedule']:
            print(f"\n{day['day']}, {day['date']}:")
            for timeslot in day['timeslots']:
                sessions_count = len(timeslot['sessions'])
                print(f"  {timeslot['time']}: {sessions_count} sessions")
                for session in timeslot['sessions']:
                    session_id = session.get('session_id', 'Unknown')
                    room = session.get('room', 'Unknown')
                    topics = ', '.join(get_session_topics(session))
                    print(f"    - {session_id} in {room}: {topics}")

        print(f"\n✓ Scheduling complete!")
        print(f"✓ Full session info saved to: {args.output}")

        # Verify author constraints are satisfied
        if author_constraints:
            print("\n" + "="*60)
            print("AUTHOR CONSTRAINT VERIFICATION")
            print("="*60)
            print("\nDemonstrating that each constraint is satisfied:\n")

            for i, constraint in enumerate(author_constraints):
                author_name = constraint.get('author_name')
                constraint_type = constraint.get('type')

                print(f"Constraint {i+1}: {author_name}")
                print("-" * 60)

                if constraint_type == 'date_constraint':
                    unavailable_dates = constraint.get('unavailable_dates', [])
                    print(f"Type: Date constraint")
                    print(f"Unavailable dates: {', '.join(unavailable_dates)}")
                elif constraint_type == 'timeslot_constraint':
                    date = constraint.get('date')
                    unavailable_timeslots = constraint.get('unavailable_timeslots', [])
                    print(f"Type: Timeslot constraint")
                    print(f"Unavailable: {date} at {', '.join(unavailable_timeslots)}")

                # Find all papers by this author and when they're scheduled
                author_papers = []
                for day in result['schedule']:
                    for timeslot in day['timeslots']:
                        for session in timeslot['sessions']:
                            for paper in session['papers']:
                                paper_authors = get_paper_authors(paper)
                                if author_name in paper_authors:
                                    author_papers.append({
                                        'id': paper['id'],
                                        'title': paper['title'],
                                        'day': day['day'],
                                        'date': day['date'],
                                        'time': timeslot['time'],
                                        'room': session.get('room', 'Unknown')
                                    })

                if author_papers:
                    print(f"\nAuthor's papers scheduled at:")
                    for paper_info in author_papers:
                        print(f"  • {paper_info['id']}: {paper_info['title'][:60]}")
                        print(f"    → {paper_info['day']}, {paper_info['date']} at {paper_info['time']} in {paper_info['room']}")

                        # Verify constraint is met
                        if constraint_type == 'date_constraint':
                            if paper_info['date'] in unavailable_dates:
                                print(f"    ✗ VIOLATION: Scheduled on unavailable date!")
                            else:
                                print(f"    ✓ Not on unavailable date")
                        elif constraint_type == 'timeslot_constraint':
                            if paper_info['date'] == date and paper_info['time'] in unavailable_timeslots:
                                print(f"    ✗ VIOLATION: Scheduled during unavailable timeslot!")
                            else:
                                print(f"    ✓ Not during unavailable timeslot")
                else:
                    print(f"\n  (No papers by {author_name} in this schedule)")

                print()

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
