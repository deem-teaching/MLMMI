# MLMMI

This repository contains the programming assignments for the course **Machine Learning Model Management and Inference**. Assignments are organized by semester.

## Semesters

- **[SS26](SS26/)** — Summer Semester 2026

## Creating a Private Copy of This Repository

GitHub does not support private forks of public repositories. To work on the assignments privately, create a **new private repo** and link this repository as an upstream remote:

```bash
# 1. Clone this repository
git clone git@github.com:deem-teaching/MLMMI.git
cd MLMMI

# 2. Rename the original remote to "upstream"
git remote rename origin upstream

# 3. Create a new **private** repository on GitHub (e.g. github.com:YOU/MLMMI.git)
#    Then add it as the new origin:
git remote add origin git@github.com:YOU/MLMMI.git
git push -u origin main
```

**Important:** Add the Teaching Assistant **[@e-strauss](https://github.com/e-strauss)** as a collaborator to your private repository so that your submissions can be graded.

To pull future updates from the course repository:

```bash
git fetch upstream
git merge upstream/main
```
