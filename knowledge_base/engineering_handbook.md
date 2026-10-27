# ACME Global - Engineering Handbook & Technical Standards

## 1. Code Review & Pull Request Standards
- **Reviewers**: Every pull request (PR) requires at least two peer approvals before merging to `main`.
- **Test Coverage**: All new features and bugfixes must include automated unit or integration tests. Overall test coverage threshold is maintained at 80% or higher.
- **PR Description**: Each PR must document:
  1. The problem being solved (referencing a Jira or GitHub ticket).
  2. The architectural approach taken.
  3. Evidence of manual verification or test output.

## 2. Release & Deployment Cadence
- **Sprints**: Engineering operates on 2-week agile sprints ending every other Wednesday.
- **Deployment Windows**: Production releases occur on Tuesday and Thursday mornings at 10:00 AM UTC.
- **Deployment Freezes**: We maintain a strict "No-Deploy Friday" policy. Emergency hotfixes on Fridays require explicit sign-off from the VP of Engineering or on-call Incident Commander.

## 3. Incident Management & Severity Levels
- **SEV-1 (Critical)**: Complete outage or critical data breach affecting > 5% of active users.
  - On-call acknowledgment SLA: Under 5 minutes.
  - Mitigation target: Under 30 minutes.
  - Post-Mortem: Blameless post-mortem document completed within 48 hours.
- **SEV-2 (Major)**: Degraded functionality for important features with no immediate workaround.
  - Acknowledgment SLA: Under 15 minutes.
- **SEV-3 (Minor)**: Cosmetic bugs or non-blocking defects scheduled for standard sprint backlog.
