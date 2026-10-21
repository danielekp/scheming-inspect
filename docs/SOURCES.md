# Source audit

Checked 2026-10-21. This is an availability audit, not a guarantee that no unindexed release exists.

| Source | Finding and use |
|---|---|
| [Paper v2](https://arxiv.org/abs/2412.04984v2), [HTML](https://arxiv.org/html/2412.04984v2) | Primary methodology. Section 4 explicitly limits reproducibility; Appendix D says the original experiments used Inspect. Table 19 identifies the private o1 checkpoint and historical public models. |
| [Apollo Research GitHub](https://github.com/ApolloResearch) | No public release of this six-evaluation suite located during the audit. Related projects should not be mistaken for this benchmark. |
| [Apollo Research Hugging Face](https://huggingface.co/apollo-research) | No corresponding full original dataset located. |
| [Author-linked transcript folder](https://drive.google.com/drive/folders/1joO-VPbvWFJcifTPwJHyfXKUzguMp-Bk) | Selected published rollouts, not the complete evaluation dataset. The paper links numbered examples in its appendix tables. Not vendored. |
| [Inspect documentation](https://inspect.aisi.org.uk/) | Framework API and execution behavior; implementation pinned to inspect-ai 0.3.281. |

The paper carries the arXiv nonexclusive distribution license, not a blanket MIT/CC-BY code/data license. No such license was located for the selected transcripts. This repository therefore uses newly written prompts and fixtures with attribution rather than copying the original scenario text or redistributing transcript PDFs.

The original arithmetic question distribution, prompts, grading code, complete 33 scenarios, and email corpus are not recoverable exactly from the paper. Replacements are deterministic and explicitly marked. The mixed-question ablation uses elementary science and cyber hygiene rather than claiming to recover the original mixed-subject dataset.

Original results cannot be recreated exactly with a different model, different data, or a modern provider alias. Obtaining the withheld materials and access to historical checkpoints is an external prerequisite for an exact empirical replication.
