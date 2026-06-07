# GECO download checklist

Use these files for the main sprint:

- EnglishMaterial
- MonolingualReadingData
- SubjectInformation

Skip these initially:

- DutchMaterials
- L1ReadingData
- L2ReadingData

Reason: the paper's main design is English monolingual reading. The bilingual and Dutch files are only useful for an optional robustness extension.

After downloading, run:

```bash
python -m src.inspect_tabular --input data/raw/geco/MonolingualReadingData.xlsx --rows 5
```

Then fill `configs/geco.yaml` with the exact column names.
