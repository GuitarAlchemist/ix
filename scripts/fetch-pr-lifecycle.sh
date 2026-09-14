#!/usr/bin/env bash
# Fetch PR lifecycle history as JSONL for memristive-markov's work-history
# model (Demerzel#596). Read-only: one paginated GraphQL query per repo.
#
# Usage: scripts/fetch-pr-lifecycle.sh [owner] [repo ...] > prs.jsonl
# Default: GuitarAlchemist ix ga tars Demerzel hari gaia
#
# One line per PR. Only lifecycle-relevant timeline items are kept:
# draft/ready flips, APPROVED reviews, merge-candidate and worker:* labels,
# merge, close (dropped when the PR merged: GitHub emits both), reopen.
# No titles, bodies or comment text.
set -euo pipefail

owner="${1:-GuitarAlchemist}"
shift || true
repos=("$@")
if [ ${#repos[@]} -eq 0 ]; then
  repos=(ix ga tars Demerzel hari gaia)
fi

query='query($owner:String!,$name:String!,$endCursor:String){
  repository(owner:$owner,name:$name){
    pullRequests(first:50,after:$endCursor,orderBy:{field:CREATED_AT,direction:ASC}){
      pageInfo{hasNextPage endCursor}
      nodes{
        number createdAt isDraft state headRefName author{login}
        timelineItems(first:100,itemTypes:[READY_FOR_REVIEW_EVENT,CONVERT_TO_DRAFT_EVENT,PULL_REQUEST_REVIEW,MERGED_EVENT,CLOSED_EVENT,REOPENED_EVENT,LABELED_EVENT]){
          nodes{
            __typename
            ... on ReadyForReviewEvent{createdAt}
            ... on ConvertToDraftEvent{createdAt}
            ... on PullRequestReview{submittedAt state}
            ... on MergedEvent{createdAt}
            ... on ClosedEvent{createdAt}
            ... on ReopenedEvent{createdAt}
            ... on LabeledEvent{createdAt label{name}}
          }
        }
      }
    }
  }
}'

for repo in "${repos[@]}"; do
  gh api graphql --paginate -F owner="$owner" -F name="$repo" -f query="$query" --jq '
    .data.repository.pullRequests.nodes[] | .state as $state | {
      repo: "'"$repo"'",
      number,
      created_at: .createdAt,
      state,
      is_draft: .isDraft,
      author: (.author.login // "ghost"),
      head_ref: .headRefName,
      events: [ .timelineItems.nodes[] |
        if .__typename == "ReadyForReviewEvent" then {kind: "ready_for_review", at: .createdAt}
        elif .__typename == "ConvertToDraftEvent" then {kind: "convert_to_draft", at: .createdAt}
        elif .__typename == "PullRequestReview" and .state == "APPROVED" then {kind: "approved", at: .submittedAt}
        elif .__typename == "MergedEvent" then {kind: "merged", at: .createdAt}
        elif .__typename == "ClosedEvent" and $state != "MERGED" then {kind: "closed", at: .createdAt}
        elif .__typename == "ReopenedEvent" then {kind: "reopened", at: .createdAt}
        elif .__typename == "LabeledEvent" and (.label.name | test("^(worker:|fleet:merge-ready$|agent-blackbox-reviewed$)")) then {kind: "labeled", at: .createdAt, label: .label.name}
        else empty end
      ]
    }'
done
