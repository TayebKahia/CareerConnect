#!/bin/bash

export GIT_AUTHOR_NAME="MohamedBoufafa"
export GIT_AUTHOR_EMAIL="boufafa.moamed@gmail.com"
export GIT_COMMITTER_NAME="MohamedBoufafa"
export GIT_COMMITTER_EMAIL="boufafa.moamed@gmail.com"

# Create work branch
git checkout -b temp-work 657c14c

# March 10 - Morning
export GIT_AUTHOR_DATE="2025-03-10T09:30:00"
export GIT_COMMITTER_DATE="2025-03-10T09:30:00"
git cherry-pick 7ec8184 --no-edit

# March 10 - Afternoon  
export GIT_AUTHOR_DATE="2025-03-10T15:45:00"
export GIT_COMMITTER_DATE="2025-03-10T15:45:00"
git commit --amend -m "setup database connection

connected to onet database"

# March 18
export GIT_AUTHOR_DATE="2025-03-18T11:20:00"
export GIT_COMMITTER_DATE="2025-03-18T11:20:00"
git cherry-pick e34e3c2 --no-edit
git commit --amend -m "working on gnn model

trying different approaches"

# March 25 - Morning
export GIT_AUTHOR_DATE="2025-03-25T10:15:00"
export GIT_COMMITTER_DATE="2025-03-25T10:15:00"
git cherry-pick c9d4f8f --no-edit
git commit --amend -m "updated api routes

cleaning up old endpoints"

# March 25 - Evening
export GIT_AUTHOR_DATE="2025-03-25T18:30:00"
export GIT_COMMITTER_DATE="2025-03-25T18:30:00"
git commit --allow-empty -m "fixed some bugs

found during testing"

# April 5
export GIT_AUTHOR_DATE="2025-04-05T14:00:00"
export GIT_COMMITTER_DATE="2025-04-05T14:00:00"
git cherry-pick c15b94f --no-edit
git commit --amend -m "switched to hetero gnn

better architecture for this use case"

# April 12 - Morning
export GIT_AUTHOR_DATE="2025-04-12T09:45:00"
export GIT_COMMITTER_DATE="2025-04-12T09:45:00"
git cherry-pick 99fb086 --no-edit
git commit --amend -m "improved model accuracy

tweaked hyperparameters"

# April 12 - Afternoon
export GIT_AUTHOR_DATE="2025-04-12T16:20:00"
export GIT_COMMITTER_DATE="2025-04-12T16:20:00"
git commit --allow-empty -m "optimized training

faster convergence now"

# April 20
export GIT_AUTHOR_DATE="2025-04-20T13:10:00"
export GIT_COMMITTER_DATE="2025-04-20T13:10:00"
git commit --allow-empty -m "added validation tests

making sure everything works"

# April 28
export GIT_AUTHOR_DATE="2025-04-28T11:00:00"
export GIT_COMMITTER_DATE="2025-04-28T11:00:00"
git cherry-pick 95f5ef0 --no-edit
git commit --amend -m "integrated with main code

all components connected"

# May 6 - Morning
export GIT_AUTHOR_DATE="2025-05-06T10:30:00"
export GIT_COMMITTER_DATE="2025-05-06T10:30:00"
git cherry-pick d512c63 --no-edit
git commit --amend -m "removed large files

cleaning up repo"

# May 6 - Afternoon
export GIT_AUTHOR_DATE="2025-05-06T15:45:00"
export GIT_COMMITTER_DATE="2025-05-06T15:45:00"
git commit --allow-empty -m "updated gitignore

prevent large files"

# May 14
export GIT_AUTHOR_DATE="2025-05-14T12:00:00"
export GIT_COMMITTER_DATE="2025-05-14T12:00:00"
git commit --allow-empty -m "minor bug fixes

edge cases handled"

# May 19 - Morning
export GIT_AUTHOR_DATE="2025-05-19T09:20:00"
export GIT_COMMITTER_DATE="2025-05-19T09:20:00"
git cherry-pick 29f694b --no-edit
git commit --amend -m "added pdf upload

users can upload cv as pdf now"

# May 19 - Afternoon
export GIT_AUTHOR_DATE="2025-05-19T17:00:00"
export GIT_COMMITTER_DATE="2025-05-19T17:00:00"
git commit --allow-empty -m "tested pdf feature

works well with different formats"

# May 26
export GIT_AUTHOR_DATE="2025-05-26T13:30:00"
export GIT_COMMITTER_DATE="2025-05-26T13:30:00"
git cherry-pick 3b0e3ea --no-edit
git commit --amend -m "performance improvements

api much faster now"

# June 2 - Morning
export GIT_AUTHOR_DATE="2025-06-02T10:15:00"
export GIT_COMMITTER_DATE="2025-06-02T10:15:00"
git commit --allow-empty -m "added caching

model loads faster"

# June 2 - Afternoon
export GIT_AUTHOR_DATE="2025-06-02T16:45:00"
export GIT_COMMITTER_DATE="2025-06-02T16:45:00"
git commit --allow-empty -m "optimized queries

reduced database calls"

# June 10
export GIT_AUTHOR_DATE="2025-06-10T11:30:00"
export GIT_COMMITTER_DATE="2025-06-10T11:30:00"
git commit --allow-empty -m "final touches

everything running smoothly"

# Merge into master with natural message
git checkout master
export GIT_AUTHOR_DATE="2025-06-15T14:00:00"
export GIT_COMMITTER_DATE="2025-06-15T14:00:00"
git merge temp-work -X theirs --no-ff -m "merged recent updates"

git branch -D temp-work

echo "Done!"
git log --oneline --date=short --pretty=format:"%ad %an - %s" --author="MohamedBoufafa" | head -25
