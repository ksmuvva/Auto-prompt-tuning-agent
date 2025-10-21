# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (cli.py)                  │
│  Interactive commands, user input, display results          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AI Agent Core (core.py)                        │
│  ┌───────────┐  ┌──────────┐  ┌────────────────┐          │
│  │  Memory   │  │ Reasoning│  │ State Manager  │          │
│  │  System   │  │  Engine  │  │                │          │
│  └───────────┘  └──────────┘  └────────────────┘          │
└───┬──────────────────┬─────────────────┬───────────────────┘
    │                  │                 │
    ▼                  ▼                 ▼
┌─────────┐    ┌──────────────┐    ┌──────────────────┐
│  Data   │    │    Prompt    │    │   LLM Service    │
│Processor│    │    Tuner     │    │                  │
│         │    │              │    │  ┌────────────┐  │
│  CSV    │    │  ┌────────┐  │    │  │  OpenAI    │  │
│  Load   │    │  │Metrics │  │    │  ├────────────┤  │
│  Filter │    │  │Eval    │  │    │  │ Anthropic  │  │
│  Stats  │    │  └────────┘  │    │  ├────────────┤  │
│         │    │              │    │  │   Mock     │  │
│         │    │  Templates   │    │  └────────────┘  │
└─────────┘    └──────────────┘    └──────────────────┘
```

## Component Interaction Flow

### 1. Initialization Flow
```
User → CLI.init() → Agent.init() → LLMService.init()
                                 → Memory.load()
                                 → TemplateLibrary.init()
```

### 2. Data Processing Flow
```
CSV Files → DataProcessor.load_csv_files()
         → DataProcessor.merge_transactions()
         → DataProcessor.validate_data()
         → DataProcessor.filter_transactions()
         → DataProcessor.prepare_for_llm()
         → Formatted Data String
```

### 3. Prompt Tuning Flow (Quick Mode)
```
User Command → Agent.run_analysis("quick")
            → PromptTuner.test_single_prompt() [3x]
            → LLMService.generate()
            → Metrics.evaluate_prompt()
            → Metrics.compare_prompts()
            → Return best prompt
```

### 4. Adaptive Tuning Flow
```
User Command → Agent.run_analysis("adaptive")
            → PromptTuner.run_adaptive_tuning()
            → Loop (max 3 iterations):
                ├─ Test all prompts
                ├─ Evaluate metrics
                ├─ Identify best prompt
                ├─ Generate improved prompt (LLM)
                ├─ Test new prompt
                └─ Check target score
            → Return final best
```

## Data Flow Diagram

```
┌────────────┐
│ 30 CSV     │
│ Files      │
└─────┬──────┘
      │
      ▼
┌────────────────────────┐
│ DataProcessor          │
│  - Load & Merge        │
│  - Validate            │
│  - Filter (>250 GBP)   │
│  - Statistical Anomaly │
│  - Ground Truth        │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Formatted Data String  │
│  - Summary stats       │
│  - Sample transactions │
│  - Max 1000 rows       │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐      ┌──────────────┐
│ Prompt Template        │◄─────┤ Template Lib │
│  {data}, {threshold}   │      └──────────────┘
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Formatted Prompt       │
│  (Complete prompt with │
│   data inserted)       │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ LLM Service            │
│  - OpenAI/Anthropic    │
│  - Generate response   │
│  - Track tokens/time   │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Raw LLM Response       │
│  (Text output)         │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Metrics Evaluator      │
│  - Parse response      │
│  - Extract data        │
│  - Compare to truth    │
│  - Calculate scores    │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Evaluation Results     │
│  - Accuracy: 0.85      │
│  - F1 Score: 0.82      │
│  - Composite: 0.84     │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Agent Decision         │
│  - Store in memory     │
│  - Update best prompt  │
│  - Generate new prompt?│
└────────────────────────┘
```

## Class Diagram

```
┌─────────────────────────────┐
│   PromptTuningAgent         │
├─────────────────────────────┤
│ - memory: AgentMemory       │
│ - llm_service: LLMService   │
│ - data_processor            │
│ - prompt_tuner              │
│ - template_library          │
├─────────────────────────────┤
│ + run_analysis()            │
│ + add_custom_prompt()       │
│ + get_recommendations()     │
│ + think()                   │
└──────────┬──────────────────┘
           │ uses
           ▼
┌─────────────────────────────┐
│     PromptTuner             │
├─────────────────────────────┤
│ - llm_service               │
│ - template_library          │
│ - metrics_evaluator         │
├─────────────────────────────┤
│ + test_single_prompt()      │
│ + test_all_prompts()        │
│ + run_adaptive_tuning()     │
│ + generate_improved_prompt()│
└──────────┬──────────────────┘
           │ uses
           ▼
┌─────────────────────────────┐
│      PromptMetrics          │
├─────────────────────────────┤
│ - metrics_history           │
├─────────────────────────────┤
│ + parse_llm_response()      │
│ + calculate_accuracy()      │
│ + calculate_f1()            │
│ + evaluate_prompt()         │
│ + compare_prompts()         │
└─────────────────────────────┘

┌─────────────────────────────┐
│      LLMService             │
├─────────────────────────────┤
│ - provider: LLMProvider     │
│ - request_history           │
├─────────────────────────────┤
│ + generate()                │
│ + batch_generate()          │
│ + switch_provider()         │
└──────────┬──────────────────┘
           │ interface
           ▼
┌─────────────────────────────┐
│     LLMProvider (ABC)       │
├─────────────────────────────┤
│ + generate()                │
│ + count_tokens()            │
└──────────┬──────────────────┘
           │ implements
           ▼
    ┌─────┴─────┬─────────┐
    ▼           ▼         ▼
┌─────────┐ ┌─────────┐ ┌────┐
│ OpenAI  │ │Anthropic│ │Mock│
└─────────┘ └─────────┘ └────┘
```

## State Management

### Agent States
```
┌──────────────┐
│ Initialized  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Data Loaded  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Tuning Active│ ◄──┐
└──────┬───────┘    │
       │            │
       ▼            │
┌──────────────┐    │
│ Evaluating   │────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Results Ready│
└──────────────┘
```

## Memory Architecture

```
┌─────────────────────────────────────┐
│         AgentMemory                 │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Short-Term Memory          │  │
│  │   - Last 50 interactions     │  │
│  │   - Volatile                 │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Long-Term Memory           │  │
│  │   - Key-value store          │  │
│  │   - Persistent (JSON)        │  │
│  │   - Timestamps               │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Learned Patterns           │  │
│  │   - Best prompts history     │  │
│  │   - Performance insights     │  │
│  │   - Persistent               │  │
│  └──────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   logs/agent_memory.json            │
└─────────────────────────────────────┘
```

## Metrics Calculation Pipeline

```
LLM Response (Text)
    │
    ▼
┌──────────────────┐
│ Parse Response   │
│  - Regex         │
│  - JSON extract  │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│ Extract Data     │
│  - Transactions  │
│  - Amounts       │
│  - IDs           │
└────┬─────────────┘
     │
     ├─────────────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌──────────────┐                    ┌──────────────┐
│  Accuracy    │                    │ Completeness │
│  Compare IDs │                    │ Check for:   │
│  vs truth    │                    │ - High value │
│              │                    │ - Anomalies  │
│ Score: 0-1   │                    │ - Summary    │
└──────────────┘                    │ Score: 0-1   │
                                    └──────────────┘
     ▼                                     ▼
┌──────────────┐                    ┌──────────────┐
│ Precision    │                    │ Format       │
│ Recall       │                    │ Quality      │
│ F1 Score     │                    │ - Structure  │
│              │                    │ - Bullets    │
│ Scores: 0-1  │                    │ - Amounts    │
└──────────────┘                    │ Score: 0-1   │
                                    └──────────────┘
     │                                     │
     │              ┌──────────────┐       │
     │              │ Specificity  │       │
     │              │ - IDs        │       │
     │              │ - Dates      │       │
     │              │ - Details    │       │
     │              │ Score: 0-1   │       │
     │              └──────┬───────┘       │
     │                     │               │
     └──────────┬──────────┴───────────────┘
                ▼
    ┌───────────────────────┐
    │  Composite Score      │
    │                       │
    │  = 0.30*Accuracy      │
    │  + 0.25*F1            │
    │  + 0.15*Completeness  │
    │  + 0.15*Format        │
    │  + 0.15*Specificity   │
    │                       │
    │  Score: 0-1           │
    └───────────────────────┘
```

## File I/O Architecture

```
┌─────────────────┐
│ Input Files     │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌──────┐ ┌──────┐   ┌──────┐   ┌──────┐
│ CSV  │ │ CSV  │...│ CSV  │   │.env  │
│  1   │ │  2   │   │  30  │   │      │
└──────┘ └──────┘   └──────┘   └──────┘

         │
         ▼
┌─────────────────────────────────────┐
│     Agent Processing                │
└────────┬────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌──────┐ ┌──────┐   ┌──────┐   ┌──────┐
│Result│ │Best  │   │Metrics   │Memory│
│ JSON │ │Prompt│   │ JSON │   │ JSON │
└──────┘ └──────┘   └──────┘   └──────┘
```

## Technology Stack Layers

```
┌─────────────────────────────────────────┐
│        User Interface Layer             │
│  - CLI (agent/cli.py)                   │
│  - Interactive REPL                     │
│  - Command processing                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Application Layer                │
│  - AI Agent Core (agent/core.py)        │
│  - Business logic                       │
│  - Orchestration                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Service Layer                    │
│  - LLM Service (llm_service.py)         │
│  - Prompt Tuner (prompt_tuner.py)       │
│  - Data Processor (data_processor.py)   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Evaluation Layer                 │
│  - Metrics System (metrics.py)          │
│  - Template Library (templates.py)      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Infrastructure Layer             │
│  - File I/O                             │
│  - Configuration                        │
│  - Memory persistence                   │
│  - Logging                              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        External Services                │
│  - OpenAI API                           │
│  - Anthropic API                        │
└─────────────────────────────────────────┘
```

This architecture ensures:
- **Modularity**: Each component is independent
- **Extensibility**: Easy to add new LLM providers
- **Testability**: Mock provider for testing
- **Maintainability**: Clear separation of concerns
- **Scalability**: Can handle large datasets
