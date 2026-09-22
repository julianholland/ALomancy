# Mission Statement — ALomancy

## Why this exists

ALomancy is a **general purpose active learning (AL) manager** for training machine-learned interatomic potentials (MLIPs). The core principle is **modularity**, built on a common AL loop skeleton. The eventual outlook is to support multiple AL schemes (committee-style, furthest-point sampling, uncertainty-based MD, etc.) as interchangeable skeletons, all calling into the same set of modules below.

## Core modules

### Initialization
How the initial training set is built. Should handle:
- Reading from an external database
- Specifying what is being trained on (target structures, e.g. surfaces)
- Optionally generating the necessary training data (dimers, trimers, MP crystals, stretch/compress, amorphization, rattles)
- Submitting all these structures to a relevant HPC for high-accuracy evaluation, and deciding whether they should be geometry-optimized first

### MLIP training
Where an MLIP is produced from the training data. Should handle:
- Training the MLIP
- Validating the MLIP
- Retrieving MLIP info (for parity/MAE/loss plots)
- Submission to an HPC machine to train (preferably with GPUs)

### Structure generation
How new structures are generated for the test/train sets. Should handle:
- Creating new structures based on what the user has marked as important, ensuring sufficient novelty
- Ensuring the structures represent a diverse sweep of what's needed
- Submission to an appropriate HPC for generation

### High-accuracy evaluation
Where training-quality data comes from. Should handle:
- Retrieving energy, forces, and (optionally) stress from the provided structures
- Either single-point or geometry-relaxation calls
- Submission to an HPC

## Design principles (apply across all modules)

- Plotting should be **live, per-loop, and informative**
- The best current model should be stored in its own directory under `results/`
- Once set up, a run should be **as hands-off as possible** — aside from unavoidable HPC sign-ins
- All structures produced and evaluated should be stored in the **global database**
- It should be **very easy to resume** from each module independently

## Current known gaps (see `TODO.md`)

- `GlobalDatabase` doesn't fully self-heal from mid-run HDF5/SQLite corruption once at least one AL loop has completed — violates the "easy to resume" principle
- Parity plots aren't populating even on fresh post-v0.4.2 runs — violates the "live, informative plotting" principle
