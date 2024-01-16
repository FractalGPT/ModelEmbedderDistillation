# ModelEmbedderDistillation [[RUS](https://github.com/FractalGPT/ModelEmbedderDistilation/blob/main/README_RUS.md)]

## About the Project
`ModelEmbedderDistillation` is a project focused on distilling complex embedding models (such as Sbert, E5) to simplify and enhance their efficiency for use in various machine learning applications.

## Features

- **Sbert Distillation**: The process of simplifying the Sbert model to optimize performance without significant loss of accuracy.
- **Sbert Distillation with Layer Decomposition**: An advanced distillation method that includes the decomposition of model layers for deeper optimization.
- **C# Version Based on AI Framework**: Implementation of the aforementioned distillation methods in C#, using the AI Framework for easy integration into .NET projects.

## Getting Started

To get started with `ModelEmbedderDistillation`, clone the repository and follow the instructions in the installation section.

## Roadmap SBert

| Stage                                         | Tasks                              | Status                |
| --------------------------------------------- | ---------------------------------- | --------------------- |
| **Sbert Distillation**                        | Researching distillation methods   | 🟢 Completed          |
| **Distillation with Layer Decomposition**     | Developing decomposition method    | 🟢 Completed         |
| **C# Version Based on [AI Framework](https://github.com/AIFramework/AIFrameworkOpen)** | Transferring algorithms to C#      | 🟢 Completed        |
|                                               | Integrating with AI Framework      | 🟢 Completed        |
| **Optimization and Expansion**                | Performance optimization           | 🔴 Not Started        |
|                                               | Supporting additional models       | 🔴 Not Started        |
| **Documentation and Examples**                | Developing documentation           | 🟡 In Progress        |
|                                               | Creating usage examples            | 🟡 In Progress       |

## Models

* Distilled SBERT [FractalGPT/SbertDistil](https://huggingface.co/FractalGPT/SbertDistil)
  * Run example in Collab: <a target="_blank" href="https://colab.research.google.com/drive/1m3fyh632htPs9UiEu4_AkQfrUtjDqIQq?usp=sharing"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* Distilled SBERT with SVD [FractalGPT/SbertSVDDistil](https://huggingface.co/FractalGPT/SbertSVDDistil)
  * Run example in Collab: <a target="_blank" href="https://colab.research.google.com/drive/1R9hHbEpyGEYO5Nw3p5VWTc-bny3PqiZs"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* Ported model for use in C# [FractalGPT/SbertDistilAIFr](https://huggingface.co/FractalGPT/SbertDistilAIFr)

## License

* This project is distributed under the [Apache 2.0 License](https://github.com/FractalGPT/ModelEmbedderDistilation/blob/main/LICENSE). See the LICENSE file for more details.
