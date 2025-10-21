# Tags JSON Format

## Description

The tags JSON file contains a curated taxonomy of topic tags used for classifying papers and clustering them into themed sessions. This is the controlled vocabulary used throughout the pipeline.

## Purpose

- Defines the official topic taxonomy for the conference
- Provides descriptions for each topic tag
- Used during paper tagging (manual or AI-assisted)
- Guides session creation by defining valid topics

## File Location

`data/tags.json`

## Structure

The file is a JSON object containing an array of tag definitions.

## Required Fields

- `tags` - Array of tag objects
  - `name` - Tag name/identifier (used in paper assignments)
  - `description` - What this tag covers

## Optional Fields

- `version` - Taxonomy version number
- `created_date` - When taxonomy was created

## Example

```json
{
  "tags": [
    {
      "name": "Software testing automation",
      "description": "Techniques and automation for verifying and validating software quality, including unit testing, integration testing, and test generation.",
    },
    {
      "name": "Program analysis",
      "description": "Static and dynamic analysis techniques for understanding program behavior, detecting bugs, and verifying properties.",
    },
    {
      "name": "Software security and privacy",
      "description": "Techniques for identifying and preventing security vulnerabilities, privacy leaks, and attacks in software systems."
    },
    {
      "name": "AI-assisted development",
      "description": "Use of machine learning and AI to assist in software development tasks including code generation, bug detection, and code review.",
    }
  ],
  "version": "1.0.0",
  "created_date": "2025-01-15",
}
```

## Notes

- Tag names should be consistent and used exactly as specified in paper assignments
- Keep descriptions clear and distinguishable from other tags
- Aim for 20-40 tags for a typical conference
