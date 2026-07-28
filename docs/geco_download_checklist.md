# GECO download checklist

For the monolingual English PACLIC analysis, obtain the official GECO files needed to verify both stimuli and reader metadata:

- `MonolingualReadingData.xlsx` — required for the main participant-word table;
- `EnglishMaterial.xlsx` — retain for stimulus cross-checks;
- `SubjectInformation.xlsx` — retain for participant/subset verification.

The current pipeline reads the main monolingual workbook and filters rows where:

```text
GROUP = monolingual
LANGUAGE_RANK = L1
LANGUAGE = English
```

Before a full run:

```bash
cd /path/to/paclic26_codebase_complete
source .venv/bin/activate
python -m src.inspect_tabular \
  --input data/raw/geco/MonolingualReadingData.xlsx \
  --sheet DATA \
  --rows 5
```

Confirm the exact columns listed in `configs/geco.yaml`. If the official workbook version uses different names, update only the YAML or GECO mapping after documenting the difference.

Do not redistribute GECO through the repository. The raw and processed data directories are ignored by Git.
