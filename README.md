# Video Understanding Agent

An experimental multimodal AI system for understanding video streams through object detection, tracking, memory, and natural language reasoning.

The goal of this project is to move beyond frame-by-frame object detection and build a system capable of answering questions about events occurring in videos.

Examples:

* "Who entered first?"
* "How many people crossed the doorway?"
* "When did the red car appear?"
* "Describe what happened during the last minute."
* "Find moments containing laptops."

---

## Overview

The project combines computer vision, vector retrieval, and language models to transform video streams into structured memories that can be queried using natural language.

Planned pipeline:

```text
Video
↓
Object Detection
↓
Multi-Object Tracking
↓
Event Extraction
↓
Scene Understanding
↓
Memory
↓
Retrieval
↓
LLM Reasoning
↓
Natural Language Answers
```

---

## Architecture

```text
Video
↓
YOLO
↓
ByteTrack
↓
Event Builder
↓
Scene Captioning
↓
Vector Embeddings
↓
Memory Store
↓
Retrieval Layer
↓
Language Model
↓
Question Answering
```

---

## Features

### Current

* Repository setup
* System design and architecture planning

### Planned

* Real-time object detection
* Multi-object tracking
* Event extraction
* Scene captioning
* Vector-based memory
* Semantic search
* Natural language querying
* Temporal reasoning
* Video summarization

---

## Tech Stack

### Computer Vision

* YOLO
* ByteTrack

### Vision-Language Models

* Qwen2.5-VL

### Embeddings

* SigLIP / CLIP

### Vector Database

* ChromaDB

### Language Models

* Llama 3
* Gemma
* Qwen

### Backend

* FastAPI

### Storage

* SQLite

---

## Repository Structure

```text
video-understanding-agent/

vision/
    detector.py
    tracker.py
    captioner.py

memory/
    event_builder.py
    vector_store.py

database/

agent/
    tools.py
    reasoner.py

backend/
    app.py

frontend/

tests/
```

---

## Development Roadmap

### Phase 1

* [ ] Object detection

### Phase 2

* [ ] Multi-object tracking

### Phase 3

* [ ] Event extraction

### Phase 4

* [ ] Region-based reasoning

### Phase 5

* [ ] Scene captioning

### Phase 6

* [ ] Vector memory

### Phase 7

* [ ] Retrieval layer

### Phase 8

* [ ] Natural language question answering

### Phase 9

* [ ] Temporal reasoning

### Phase 10

* [ ] Anomaly detection

---

## Motivation

Most object detection systems answer:

> "What objects are present?"

This project aims to answer:

> "What happened?"

---

## Status

Early-stage project under active development.

No benchmark results or performance metrics are reported yet. Metrics and evaluations will be added as features are implemented and tested.

---

## License

MIT
