# Task Guides

Each guide describes one SURE-EVAL task, including supported metrics, concrete
pipeline IDs, pipeline nodes, input/output formats, and CLI examples.

Task guides use canonical reported metrics (`metric`) first. When a task needs
a compatibility selector or method selector, the guide lists that separately
from the concrete `pipeline_id`. Generated descriptions also expose relative
`task_config_path` / `route_config_path`, `script_entrypoint`, and `executor`
fields so users and agents can trace a route back to implementation code.

For a machine-readable catalog of every metric → pipeline → node mapping, see [Pipeline Catalog](../pipeline_catalog.md).

- [ASR — Automatic Speech Recognition](./asr.md)
- [S2TT — Speech-to-Text Translation](./s2tt.md)
- [SD — Speaker Diarization](./sd.md)
- [SA-ASR — Speaker-Aware ASR](./sa_asr.md)
- [TTS — Text-to-Speech](./tts.md)
- [VC — Voice Conversion](./vc.md)
- [SE — Speech Enhancement](./se.md)
- [TSE — Target Speaker Extraction](./tse.md)
- [Classification / SER / GR](./classification.md)
- [KWS — Keyword Spotting](./kws.md)
- [SLU — Spoken Language Understanding](./slu.md)
