## Project Structure


`/speaker_extraction`: Scripts to generate 2-speaker mixture for speaker extraction.

`/synchronization_classification`: Scripts to generate 2-speaker mixture list for the synchronization classification pretraining.


## Dataset File Structure


├── voxceleb2
│   ├── orig
│   │   ├── train
│   │   └── test

    .
    ├── ...
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...