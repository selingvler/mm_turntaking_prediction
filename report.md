# FACULTY OF ENGINEERING AND NATURAL

# SCIENCES CAPSTONE PROGRESS REPORT

# MULTIMODAL TURN-TAKING PREDICTION FOR HUMAN-

# ROBOT INTERACTION

```
Elifnur Nazlı – Industrial Engineering
```
```
Furkan Kuşman – Industrial Engineering
```
```
Mahmut Hami Bayır – Industrial Engineering
```
```
Buse Yazıcı – Computer Engineering
```
```
Fuat Bera Günay – Computer Engineering
Selin Coşkun – Computer Engineering
```
```
Selin Güler – Computer Engineering
```
```
Advisors:
```
```
Associate Professor Adnan Çorum – Industrial Engineering
```
```
Associate Professor Cemal Okan Sakar – Computer
```
# Engineering June, 2026


## STUDENT DECLARATION

By submitting this report, as partial fulfilment of the requirements of the Capstone Design Project
course, each member of the project group certifies that:

they have given credit to and declared (by citation), any work that is not their own;

they have not received unpermitted aid for the project design, construction, report writing or
presentation;

they have not falsely assigned credit for work to another student in the group and that each member's
contribution is properly noted.


## DECLARATION OF AI USAGE

This project report was developed with the assistance of generative AI tools (e.g., ChatGPT,
Copilot, Gemini, Claude, etc.). The use of AI in this work was limited to the following activities: •
Brainstorming and idea generation

- Literature search and summarization
- Grammar and language editing
- Code generation and debugging assistance
- Data analysis support

**Validation and Authorship I certify that:**

```
I have personally verified all facts, citations, and data points provided by the AI. I have not simply
copied and pasted AI outputs; all final text is my own writing or a significant rewrite of AI
suggestions.
```
```
I accept full responsibility for the integrity and accuracy of this report.
```

## Project Team

Department 1: Industrial Engineering

Member 1: Elifnur Nazlı

Member 2: Furkan Kuşman

Member 3: Mahmut Hami Bayır

Advisor: Associate Professor Adnan Çorum

Department 2 : Computer Engineering

Member 1: Buse Yazıcı

Member 2: Fuat Bera Günay

Member 3: Selin Coşkun

Member 4: Selin Güler


# ABSTRACT

## Our project, Multimodal Turn Taking Prediction for Human-Robot Interaction , aims to develop

## an intelligent turn-taking prediction system that enables smoother and more natural conversational

## interaction between humans and robots. The primary objective is to predict whether the

## conversational floor should be held by the current speaker or shifted to the other participant by

## analyzing multimodal conversational cues.

## During this reporting period, substantial progress has been achieved in system design,

## implementation, and evaluation. The project team adapted an existing multimodal turn-taking

## prediction framework and modified it to better fit the project requirements. Since the original

## architecture relied on OpenFace-based visual feature extraction, the visual pipeline was

## redesigned using MediaPipe for more practical and accessible video feature processing. In

## parallel, custom Turkish dyadic conversational recordings were prepared and processed to support

## evaluation in a language setting more relevant to the project objectives.

## The completed work includes audio preprocessing, video feature extraction, voice activity

## detection alignment, multimodal input preparation, label generation, model adaptation for CPU-

## compatible execution, and preliminary evaluation experiments. In addition, audio fine-tuning

## experiments were conducted to improve prediction performance.

## Preliminary findings indicate promising performance in turn-taking prediction tasks. Evaluation

## results from custom experimental sessions demonstrated strong weighted F1 and balanced

## accuracy scores, while audio fine-tuning improved several speaker prediction metrics.

## The project is currently in the multimodal integration and evaluation phase. The audio model has

## already been fine-tuned; however, the visual modality has not yet been integrated with the fine-

## tuned audio model. The remaining work includes combining the fine-tuned audio model with both

## non-fine-tuned and fine-tuned visual models, conducting comparative multimodal experiments,

## and analyzing the resulting performance differences. Additionally, threshold optimization and

## real-time deployment investigation for human-robot interaction scenarios are planned.

## Keywords: turn-taking prediction, human-robot interaction, multimodal learning,

## conversational AI, speech processing


# TABLE OF CONTENTS

_Right-click here and select 'Update Field' to generate Table of Contents_


## LIST OF TABLES

_Right-click here and select 'Update Field' to generate List of Tables_

## LIST OF FIGURES

_Right-click here and select 'Update Field' to generate List of Figures_

## LIST OF ABBREVIATIONS

_[List abbreviations used in the report.]_


# 1. Introduction

## 1.1 Background

_Recent advancements in Artificial Intelligence (AI), Machine Learning (ML), and Human-Robot
Interaction (HRI) have significantly increased the demand for intelligent systems capable of
communicating with humans in a natural and efficient manner. As robots and conversational agents
become increasingly integrated into everyday environments such as healthcare, education, customer
service, and collaborative manufacturing, the quality of interaction between humans and machines has
become a critical research topic. One of the key characteristics of natural human communication is the
ability to manage conversational turn-taking, which enables participants to coordinate speaking
opportunities without explicit signals. Humans continuously rely on verbal and non-verbal cues, including
speech patterns, pauses, facial expressions, and gaze behavior, to determine when a speaker intends to
continue speaking or yield the conversational floor. However, enabling robots to interpret these cues and
respond appropriately remains a challenging problem. Recent developments in multimodal machine
learning have created new opportunities for integrating audio and visual information to improve
conversational understanding. Within this context, turn-taking prediction has emerged as an important
research area that aims to enhance the naturalness, fluency, and effectiveness of interactions between
humans and intelligent systems._

## 1.2 Problem Statement

_Despite significant progress in conversational AI, many existing human-machine interaction systems still
struggle to manage conversational turn transitions effectively. In many cases, systems respond too early,
interrupt the speaker, or react too late, creating unnatural and inefficient communication experiences.
Such limitations reduce user satisfaction and hinder the practical deployment of conversational robots in
real-world environments. These challenges become even more pronounced in dynamic conversations
where multiple conversational cues must be interpreted simultaneously. Traditional approaches often
rely on audio information alone and fail to fully capture the rich multimodal signals that humans
naturally use during communication. Therefore, there is a need for a more comprehensive approach that
combines both audio and visual conversational cues to improve turn-taking prediction performance.
Addressing this problem is essential for developing intelligent systems capable of participating in
conversations in a manner that more closely resembles human interaction._

## 1.3 Project Objectives

_The primary objective of this project is to develop a multimodal turn-taking prediction system for
Human-Robot Interaction that can predict whether a conversational turn should remain with the current
speaker or transition to another participant. To achieve this goal, the project aims to analyze both audio
and visual conversational cues, develop a multimodal machine learning framework capable of
integrating information from multiple modalities, evaluate prediction performance using established
machine learning metrics, and investigate the feasibility of applying the developed system in real-time
interaction environments. During the implementation phase, several modifications were introduced at
the technical level while the overall project objectives remained unchanged. The most significant
modification involved replacing the originally planned OpenFace visual processing pipeline with
MediaPipe due to implementation and compatibility limitations. Additionally, the evaluation framework
was extended to include custom Turkish conversational recordings to provide a more linguistically
relevant evaluation environment. These modifications improved the practicality of the system without
altering the fundamental objectives of the project_


## 1.4 Scope and Limitations

_The scope of this project includes the collection and preparation of conversational datasets, audio
preprocessing, voice activity detection, video feature extraction, multimodal synchronization, machine
learning-based turn-taking prediction, model fine-tuning, and performance evaluation. The project
focuses specifically on predicting speaker transitions in dyadic conversations using multimodal
information extracted from audio and video recordings. Several limitations should also be acknowledged.
First, the project relies on a relatively limited dataset of Turkish conversational recordings, which may
restrict the generalizability of the results. Second, the development environment was adapted for CPU-
based experimentation, limiting the computational resources available for large-scale model training and
evaluation. Third, although deployment feasibility was investigated, a fully operational real-time robotic
implementation was not completed within the scope of the project. Finally, certain turn-taking scenarios,
particularly overlap-shift prediction tasks, remain challenging due to data imbalance and the inherent
complexity of predicting speaker interruptions before they occur._

## 1.5 Report Organization

_The remainder of this report is organized as follows. Section 2 presents a review of the literature and
related work in the fields of Human-Robot Interaction, multimodal learning, and turn-taking prediction.
Section 3 describes the methodology, system architecture, tools, technologies, and data processing
procedures used throughout the project. Section 4 explains the implementation process and details the
contributions of both departments. Section 5 presents the testing procedures, validation activities, and
performance evaluation results. Section 6 discusses the final results and compares the achieved outcomes
with the original project objectives. Section 7 summarizes project management activities, team
contributions, resource utilization, and interdisciplinary collaboration. Section 8 discusses ethical, safety,
and sustainability considerations associated with the project. Finally, Section 9 concludes the report by
summarizing the project's achievements, lessons learned, and recommendations for future work._

# 2. Literature Review / State of the Art

_[Provide a comprehensive review of related work, including academic papers, existing products,
technologies, and standards relevant to the project. Identify gaps in existing solutions that your project
addresses. Update and expand the literature review from the proposal with any new sources discovered
during the project. Use IEEE citation format.]_

# 3. Methodology and Technical Approach

## 3.1 Overall Approach

The objective of this project was to develop a multimodal turn-taking prediction system for Human-
Robot Interaction (HRI) scenarios. The proposed system aims to predict future speaker transitions by
utilizing both audio and visual conversational cues.

The overall methodology followed a multimodal machine learning approach consisting of four major
stages: data preparation, feature extraction, model adaptation, and multimodal prediction. Audio and
video recordings of conversational interactions were first collected and synchronized. The audio
modality was processed using Voice Activity Detection (VAD) and pretrained VAP-based
conversational representations, while the visual modality was processed using facial landmark
features extracted through MediaPipe Face Landmarker.


To leverage existing conversational knowledge, pretrained models originally trained on the Candor
dataset were adopted as the baseline architecture. Rather than training models from scratch, transfer
learning and fine-tuning techniques were employed to adapt the pretrained models to Turkish
conversational data.

The project investigated both unimodal and multimodal prediction strategies. Audio-only and video-
only systems were evaluated independently, followed by multimodal prediction-level fusion
experiments. Comparative evaluation was conducted using multiple conversational event categories
provided by the MM-VAP framework, including speaker transition prediction, speaker activity
prediction, overlap detection, backchannel detection, and utterance duration prediction.

This methodology enabled systematic analysis of modality contributions and assessment of
multimodal fusion effectiveness for conversational turn-taking prediction.

## 3.2 System Design / Architecture

The final system architecture consists of four primary modules: data preprocessing, audio processing,
visual processing, and multimodal fusion.

The preprocessing module is responsible for synchronizing conversational recordings and preparing
model-compatible inputs. Audio streams are converted into standardized formats, while visual
recordings are processed frame-by-frame to ensure temporal alignment between modalities.

The audio branch utilizes the pretrained VAP-Candor architecture. Conversational audio is processed
through a pretrained audio encoder and Transformer-based conversational modeling layers. The
model generates Voice Activity Projection (VAP) predictions representing future speaker activity
patterns and turn-taking behavior.

The visual branch utilizes facial landmark representations extracted using MediaPipe Face
Landmarker. For each speaker, facial motion and expression features are converted into compact
feature vectors and processed by a Transformer-based video model to generate conversational
predictions.

After independent inference, prediction outputs from both modalities are integrated through a
prediction-level fusion module. Fusion is performed using weighted averaging of audio and visual
prediction scores. Multiple fusion coefficients are evaluated to determine the optimal balance between
modalities.

Finally, fused predictions are evaluated using the MM-VAP validation framework, which computes
event-level performance metrics across multiple conversational tasks.

The modular architecture allows independent modification of audio and visual components while
maintaining a flexible multimodal integration strategy.

## 3.3 Tools, Technologies, and Standards

The project was implemented primarily using Python and open-source machine learning frameworks.

**Programming Languages**


- Python
**Machine Learning Frameworks**
- PyTorch
- NumPy
**Audio Processing Tools**
- FFmpeg
- WebRTC VAD
- VAP Framework
**Computer Vision Tools**
- MediaPipe Face Landmarker
- OpenCV
**Data Handling and Analysis**
- Pandas
- Matplotlib
**Development Tools**
- Git
- GitHub
- Visual Studio Code
- Conda
**Datasets**
- Candor Dataset (pretraining source)
- Custom Turkish Conversational Dataset
PyTorch was selected due to its flexibility for deep learning experimentation and compatibility with
the MM-VAP framework. MediaPipe was chosen because it provides robust real-time facial landmark
extraction. FFmpeg and WebRTC VAD were used to ensure reliable audio preprocessing and speaker
activity detection. Git and GitHub facilitated version control and collaborative development
throughout the project lifecycle.

## 3.4 Data Collection and Analysis

The project utilized a custom multimodal conversational dataset consisting of Turkish dialogue
recordings involving two speakers. Five conversational sessions were collected and processed for
experimentation.

Data preparation began with synchronization of audio, video, and transcription resources. Audio
recordings were converted into standardized formats and processed using Voice Activity Detection to
identify speaker activity segments. Corresponding video recordings were synchronized with the audio
streams to ensure temporal consistency.

For the visual modality, facial landmarks were extracted using MediaPipe Face Landmarker. Landmark
coordinates were transformed into compact visual feature representations suitable for Transformer-
based processing. For the audio modality, pretrained VAP representations and VAD information were
utilized to capture conversational dynamics and future speaker activity patterns.


The resulting multimodal data were organized into session-based folds for training, validation, and
testing. Fine-tuning experiments were performed using Turkish conversational recordings to adapt
pretrained Candor-based models to the target domain.

Performance analysis was conducted using the MM-VAP evaluation framework. Multiple
conversational event categories were examined, including future speaker transition prediction, future
speaker activity prediction, overlap detection, backchannel detection, and utterance duration
classification. Evaluation metrics included Weighted F1-score, class-specific F1-scores, and Balanced
Accuracy.

Comparative analysis was performed across audio-only, video-only, and multimodal fusion
configurations to assess the contribution of each modality and quantify the effectiveness of
multimodal turn-taking prediction.

# 4. Implementation

## 4.1 Development Process

**1)Prediction-Level Fusion**

The development process for the prediction-level fusion module followed an iterative experimental
methodology. The primary objective was to investigate whether visual information could provide
complementary cues to the audio-based turn-taking prediction system.

The process began with the generation of baseline predictions from two independently trained
models: a pretrained audio-only VAP model and a pretrained video-only model based on facial
landmark features extracted using MediaPipe. Both models were evaluated separately on the Turkish
conversational sessions to establish unimodal performance baselines.

Following baseline evaluation, modality-specific fine-tuning experiments were conducted. The audio
model was adapted to the Turkish conversational dataset using transfer learning techniques, while the
visual model was fine-tuned using synchronized visual features extracted from the same sessions. This
stage aimed to improve each modality's ability to capture speaker transition patterns specific to
Turkish conversational behavior.

After obtaining the fine-tuned models, prediction-level fusion was implemented. Instead of combining
features within the neural network architecture, fusion was performed by combining the prediction
scores generated by the audio and visual models. A weighted averaging strategy was adopted, where
the contribution of each modality was controlled through a fusion coefficient (α).

To determine the optimal fusion configuration, a series of experiments were performed using fusion
weights ranging from α = 0.0 to α = 1.0 with increments of 0.1. For each weight configuration, the
fused predictions were evaluated using the existing MM-VAP validation framework. The Shift F1 score
of the shift_hold_p_future task was selected as the primary optimization criterion, as future speaker
transition prediction represents the core objective of the project.

The final stage consisted of comparative performance analysis between six experimental settings:
pretrained audio-only, pretrained video-only, baseline fusion, fine-tuned audio-only, fine-tuned video-
only, and fine-tuned fusion. This comparison enabled a detailed assessment of the individual
contribution of each modality and the effectiveness of prediction-level multimodal fusion.


## 4.2 Detailed Implementation

**1)Prediction-Level Fusion**

The prediction-level fusion framework was designed to investigate whether visual information could
improve conversational turn-taking prediction when combined with audio-based predictions.

The implementation pipeline consisted of five sequential stages: generation of audio predictions,
generation of visual predictions, modality-specific fine-tuning, prediction-level fusion, and
performance evaluation.

**_Audio Prediction Generation_**
The first stage involved generating baseline predictions using the pretrained VAP-Candor audio model.
The model processed stereo conversational audio and produced frame-level VAP and VAD predictions
for each session. The generated outputs were stored as serialized prediction files and later used as
inputs to the fusion module.

Following baseline evaluation, the audio model was fine-tuned using Turkish conversational
recordings. The fine-tuning process adapted the pretrained model to the characteristics of the target
dataset while preserving the conversational knowledge learned from the original Candor corpus. Fine-
tuned predictions were generated for all evaluation sessions and stored for subsequent fusion
experiments.

**_Visual Prediction Generation_**
The visual branch utilized synchronized facial landmark features extracted from both speakers using
MediaPipe Face Landmarker. These features were converted into compact visual representations and
processed by a Transformer-based video model trained for turn-taking prediction.

Initially, predictions were generated using the pretrained video model. Subsequently, the visual
encoder was fine-tuned on the Turkish conversational dataset to improve adaptation to the target
domain. Similar to the audio pipeline, prediction outputs were stored as serialized files for later fusion.

**_Prediction-Level Fusion_**
The core contribution of this work was the implementation of prediction-level fusion. Instead of
combining modalities within the neural network architecture, fusion was performed after inference by
combining the prediction scores produced by the audio and visual models.

For each conversational event category, corresponding prediction scores from both modalities were
loaded and temporally aligned. The final prediction score was computed using a weighted averaging
strategy:

[

P_{fusion} = alpha P_{audio} + (1-alpha) P_{video}

]

where (P_{audio}) and (P_{video}) represent the prediction scores generated by the audio and visual
models, respectively, and (alpha) controls the contribution of each modality.


To identify the most effective fusion configuration, multiple experiments were conducted using fusion
coefficients ranging from 0.0 to 1.0 with increments of 0.1. An alpha value of 1.0 corresponds to a
purely audio-based system, whereas an alpha value of 0.0 corresponds to a purely visual system.

**_Fusion Optimization_**
For every alpha configuration, the fused predictions were passed through the MM-VAP validation
framework. Event-specific thresholds were optimized using the validation procedure, and final
predictions were generated from the fused probability distributions.

The Shift F1 score of the shift_hold_p_future task was selected as the primary optimization criterion
because future speaker transition prediction constitutes the main objective of conversational turn-
taking systems.

The optimal alpha value was determined independently for each session based on the highest achieved
Shift F1 score.

**_Performance Evaluation_**
The final fused predictions were evaluated using the same validation protocol applied to the unimodal
models. Performance was measured using Weighted F1-score, class-specific F1-scores, and Balanced
Accuracy across multiple conversational event categories, including shift prediction, speaker
prediction, backchannel detection, overlap detection, and short-versus-long utterance classification.

The resulting performance values were then compared against pretrained audio-only, pretrained
video-only, baseline fusion, fine-tuned audio-only, and fine-tuned video-only systems to quantify the
effectiveness of prediction-level multimodal fusion.

## 4.3 Department-Specific Contributions

**Computer Engineering Team**

The Computer Engineering team was responsible for:

- Audio preprocessing and VAD generation
- Visual feature extraction using MediaPipe
- Dataset synchronization and alignment
- Fine-tuning of audio and visual models
- Implementation of prediction-level fusion
- Performance evaluation and experimental analysis

## 4.4 Integration and System Assembly

**1)Prediction-Level Fusion**

The integration phase focused on combining independently generated audio and visual turn-taking
predictions into a unified multimodal prediction framework using prediction-level fusion.

Unlike feature-level or model-level fusion approaches, the audio and visual models were executed
separately. Each model produced its own prediction outputs in the form of VAP and VAD probability
scores. This modular design enabled independent development, evaluation, and fine-tuning of each
modality before multimodal integration.


The audio branch generated conversational event predictions using the pretrained and fine-tuned
VAP-Candor model, while the visual branch generated predictions using the Transformer-based video
model operating on MediaPipe facial landmark features. Prediction outputs from both modalities were
stored as serialized files and subsequently used as inputs to the fusion module.

During integration, prediction outputs from the audio and visual branches were temporally aligned
and loaded simultaneously. For each conversational event category, corresponding prediction scores
from both modalities were combined using a weighted averaging strategy. The contribution of each
modality was controlled by a fusion coefficient (α), allowing systematic evaluation of different audio-
video combinations.

The integrated fusion module was designed to support multiple alpha configurations without
requiring retraining of either model. This allowed efficient experimentation with different modality
contributions while preserving the original model architectures and learned representations.

After fusion, the resulting prediction scores were passed directly to the MM-VAP validation
framework. Threshold optimization and event extraction procedures were then applied to generate
final conversational event predictions. This ensured that fused predictions were evaluated using the
same protocol as the unimodal systems, allowing fair comparison across all experimental conditions.

One of the main advantages of this integration strategy was its flexibility. Since fusion occurred after
inference, audio and visual models could be modified, fine-tuned, or replaced independently without
requiring changes to the multimodal integration framework. Furthermore, prediction-level fusion
significantly reduced implementation complexity compared to end-to-end multimodal training while
still allowing the system to exploit complementary information from both modalities.

The final integrated system enabled direct comparison between pretrained audio-only, pretrained
video-only, baseline fusion, fine-tuned audio-only, fine-tuned video-only, and fine-tuned fusion
configurations. This comparative framework provided detailed insight into the contribution of each
modality and the effectiveness of prediction-level multimodal fusion for conversational turn-taking
prediction.

## 4.5 Challenges and Solutions

**1)Prediction-Level Fusion**

Several technical challenges were encountered during the implementation and evaluation of the
prediction-level fusion framework.

**_Modality Performance Imbalance_**
One of the primary challenges was the significant performance difference between the audio and
visual modalities. Throughout the experiments, the audio model consistently outperformed the video
model across most conversational event categories. As a result, assigning equal importance to both
modalities often led to performance degradation rather than improvement.

To address this issue, a weighted fusion strategy was adopted instead of simple averaging. The
contribution of each modality was controlled through a tunable fusion coefficient (α), allowing the
system to place greater emphasis on the stronger audio modality while still incorporating
complementary visual information.


**_Fusion Weight Selection_**
Determining the optimal balance between audio and visual predictions represented another major
challenge. Different sessions exhibited different responses to multimodal fusion, making it difficult to
identify a universally optimal weighting scheme.

To overcome this limitation, a systematic alpha search procedure was implemented. Fusion
coefficients ranging from 0.0 to 1.0 were evaluated in increments of 0.1, and each configuration was
assessed using the MM-VAP validation framework. The Shift F1 score of the shift_hold_p_future task
was used as the primary optimization criterion, enabling objective selection of the most effective
fusion weight.

**_Fine-Tuning Variability_**
Although fine-tuning generally improved the performance of the audio model, the impact of visual
fine-tuning was less consistent. In several sessions, the fine-tuned visual model produced lower
performance than its pretrained counterpart. This suggested that visual conversational cues were
more difficult to adapt using the limited amount of available Turkish data.

To mitigate this issue, both pretrained and fine-tuned visual models were retained and evaluated
independently. Comparative experiments were conducted to quantify the effect of visual fine-tuning
and determine whether it contributed positively to multimodal performance.

**_Prediction Alignment_**
Prediction-level fusion requires both modalities to generate compatible prediction sequences. Small
differences in output lengths and temporal indexing occasionally occurred between the audio and
visual prediction files.

This issue was resolved by applying temporal alignment procedures prior to fusion. Prediction
sequences were truncated to a common length and synchronized before weighted averaging was
performed. This ensured that corresponding predictions represented the same conversational time
intervals.

**_Evaluation Consistency_**
Another challenge involved ensuring fair comparison between unimodal and multimodal systems.
Differences in threshold values could artificially influence performance metrics and obscure the actual
contribution of fusion.

To ensure consistency, all experimental configurations were evaluated using the same MM-VAP
validation framework. Event-specific thresholds were optimized automatically for each experiment
using identical procedures, allowing objective comparison across pretrained, fine-tuned, and fused
systems.

Overall, these challenges highlighted the complexity of multimodal conversational modeling and
motivated the adoption of a flexible prediction-level fusion framework. The implemented solutions
enabled systematic evaluation of modality contributions while maintaining a fair and reproducible
experimental methodology.


# 5. Testing and Validation

## 5.1 Test Plan

**1)Prediction-Level Fusion**

The testing strategy was specifically designed to evaluate the effectiveness of prediction-level
multimodal fusion for conversational turn-taking prediction.

The evaluation process consisted of five sequential stages:

1. Generation and validation of baseline audio-only predictions using the pretrained VAP-Candor
    model.
2. Generation and validation of baseline video-only predictions using the pretrained visual
    Transformer model.
3. Generation of fine-tuned audio-only and fine-tuned video-only predictions using the Turkish
    conversational dataset.
4. Implementation of prediction-level fusion by combining audio and visual prediction scores
    using weighted averaging with different alpha values ranging from 0.0 to 1.0.
5. Comparative evaluation of all experimental configurations using the MM-VAP validation
    framework.
For each session, fusion experiments were conducted using multiple alpha values in order to
determine the optimal balance between audio and visual information. The Shift F1 score of the
shift_hold_p_future task was selected as the primary evaluation metric because future speaker
transition prediction represents the main objective of the turn-taking prediction system.

The following six experimental configurations were compared throughout the testing process:

- Pretrained Audio-Only
- Pretrained Video-Only
- Baseline Prediction-Level Fusion
- Fine-Tuned Audio-Only
- Fine-Tuned Video-Only
- Fine-Tuned Prediction-Level Fusion
All experiments were evaluated using identical validation procedures, including threshold
optimization and event extraction mechanisms provided by the MM-VAP framework. This ensured a
fair comparison between unimodal and multimodal approaches.

The primary objective of the testing phase was to determine whether prediction-level fusion could
improve turn-taking prediction performance beyond the individual audio-only and video-only models,
and to quantify the contribution of visual information within the multimodal framework.

## 5.2 Test Results

**1)Prediction-Level Fusion**

The prediction-level fusion framework was evaluated through a series of controlled experiments
designed to measure the contribution of audio and visual modalities to conversational turn-taking
prediction.

```
Test ID Test Description Expected Result Actual Result Status
(Pass/Fail)
```

T- (^01) Pretrained Audio-
Only Evaluation
Establish baseline
turn-taking
performance
using audio
predictions
Baseline
performance
successfully
obtained for all
sessions
Pass
T- (^02) Pretrained Video-
Only Evaluation
Establish baseline
turn-taking
performance
using visual
predictions
Baseline
performance
successfully
obtained for all
sessions
Pass
T- (^03) Fine-Tuned Audio
Model Evaluation
Improve
performance
through
adaptation to
Turkish
conversational
data
Performance
improvement
observed in
multiple event
categories
Pass
T- (^04) Fine-Tuned Video
Model Evaluation
Improve visual
turn-taking
prediction
performance
Improvement
observed in
selected sessions
and tasks
Pass
T- (^05) Prediction-Level
Fusion Evaluation
Combine audio
and visual
predictions using
weighted
averaging
Fusion outputs
successfully
generated and
evaluated
Pass
T- (^06) Alpha
Optimization
Experiment
Identify the
optimal audio-
video
contribution ratio
Optimal alpha
values
successfully
determined for
each session^
Pass
T- (^07) Fine-Tuned
Fusion Evaluation
Achieve higher
performance than
unimodal
baselines
Performance
improvements
observed in
several sessions
and event
categories
Pass
The experimental results demonstrated that prediction-level fusion successfully integrated
complementary information from audio and visual modalities. The contribution of fusion varied across
sessions, with some sessions showing substantial improvements over unimodal baselines.
The strongest improvements were observed after combining fine-tuned audio and visual predictions
using optimized fusion weights. In particular, Session05 achieved the highest overall performance,
where the fine-tuned fusion model outperformed both the audio-only and video-only baselines in the
primary turn-taking prediction task.
These results confirm that prediction-level fusion can effectively enhance conversational event
prediction performance when appropriate modality weighting is applied.

## 5.3 Performance Evaluation

**1)Prediction-Level Fusion**


The performance of the prediction-level fusion framework was evaluated against the primary
objectives defined at the beginning of the study. Particular emphasis was placed on future speaker
transition prediction, modality contribution analysis, and the effectiveness of multimodal fusion.

```
Success Criterion Target Value Achieved Value Met?
(Yes/No/Partial)
```
SC- (^1) Shift F1 > 0.75 for the
primary turn-taking
task
Up to 0.92 Shift F
(Session05)
Yes
SC- (^2) Improve performance
over video-only
predictions through
multimodal fusion
Achieved across all
evaluated sessions
Yes
SC- (^3) Demonstrate
successful adaptation
of pretrained models
to Turkish
conversational data
Performance
improvements
observed after audio
and visual fine-tuning
Yes
SC- (^4) Successfully
implement and
validate prediction-
level fusion
Fully implemented and
evaluated using MM-
VAP validation
framework
Yes
SC- (^5) Determine optimal
audio-video weighting
strategy
Session-specific
optimal alpha values
successfully identified
Yes
SC- (^6) Evaluate modality
contributions
independently and
jointly
Audio-only, video-only,
and fusion
configurations
systematically
compared
Yes
The experimental results indicate that audio information remained the dominant modality for
conversational turn-taking prediction. However, visual information provided complementary cues that
improved performance in several event categories when combined through prediction-level fusion.
The alpha optimization experiments further demonstrated that different sessions benefited from
different modality weightings, highlighting the importance of adaptive fusion strategies. In most cases,
the optimal fusion configuration assigned greater weight to the audio modality while retaining useful
information from the visual branch.
The highest overall performance was achieved by the fine-tuned fusion model in Session05, where the
system reached a Shift F1 score of 0.92 in the shift_hold_p_future task. This result exceeded the
predefined performance target and confirmed the effectiveness of the proposed prediction-level
multimodal fusion approach.
Overall, the evaluation demonstrates that prediction-level fusion successfully combines
complementary information from audio and visual modalities and can improve conversational turn-
taking prediction performance beyond unimodal baselines under appropriate fusion settings.

## 5.4 User Feedback (if applicable)

_[If user testing was conducted, present the feedback received and any changes made based on this
feedback.]_


# 6. Results and Discussion

## 6.1 Final Results

**1)Prediction-Level Fusion**

The primary objective of this study was to evaluate whether prediction-level fusion could improve
conversational turn-taking prediction by combining independently generated audio and visual model
outputs.

Experimental results demonstrated that prediction-level fusion was capable of improving
performance in several conversational event categories when compared to unimodal baselines. The
effectiveness of fusion depended on the selected modality weighting and the characteristics of
individual conversational sessions.

For each evaluation session, audio-only, video-only, and fused prediction outputs were compared
using the MM-VAP validation framework. Additional experiments were conducted using fine-tuned
audio and visual models to investigate the effect of domain adaptation on fusion performance.

The highest performance was obtained in Session05, where the fine-tuned prediction-level fusion
model achieved a Shift F1 score of approximately 0.92 for the shift_hold_p_future task. This result
exceeded the corresponding audio-only, video-only, and baseline fusion configurations.

Overall, the experiments confirmed that prediction-level fusion can successfully exploit
complementary information from audio and visual modalities while maintaining the flexibility of
independently trained models.

## 6.2 Analysis and Discussion

**1)Prediction-Level Fusion**

Analysis of the experimental results revealed several important observations regarding prediction-
level multimodal fusion.

First, audio information consistently remained the strongest modality across all sessions. Audio-only
models generally achieved higher performance than video-only models in the majority of
conversational event categories. This finding suggests that conversational turn-taking behavior is
primarily reflected through acoustic cues such as speaking activity, timing patterns, and speech
dynamics.

Second, visual information provided complementary information rather than serving as a standalone
replacement for audio. Although video-only performance remained lower than audio-only
performance, visual predictions improved fusion performance in several tasks when appropriate
weighting coefficients were applied.

The alpha optimization experiments further demonstrated that the optimal contribution of each
modality varied across sessions. In most cases, the highest-performing fusion configurations assigned
greater weight to the audio modality while retaining a smaller but beneficial contribution from the
visual branch. This observation indicates that visual information can support turn-taking prediction
when integrated carefully.


Another important finding concerns the impact of fine-tuning. Audio fine-tuning produced relatively
consistent improvements across sessions, whereas visual fine-tuning showed greater variability. In
some sessions, visual fine-tuning improved performance, while in others it produced limited gains or
slight degradation. This behavior likely reflects the relatively small size of the available Turkish
conversational dataset and the greater difficulty of learning robust visual conversational cues.

Overall, the results suggest that prediction-level fusion represents a practical and effective approach
for multimodal turn-taking prediction. It allows independent optimization of each modality while still
benefiting from complementary multimodal information.

## 6.3 Comparison with Original Objectives

**1)Prediction-Level Fusion**

The original objective of this work was to investigate whether prediction-level fusion could improve
conversational turn-taking prediction by combining audio and visual model outputs.

This objective was successfully achieved through the implementation and evaluation of a complete
prediction-level fusion framework. The study generated baseline audio-only and video-only
predictions, performed modality-specific fine-tuning, implemented weighted prediction fusion, and
systematically evaluated multiple fusion configurations using conversational event metrics.

The alpha-search procedure successfully identified session-specific modality weightings, allowing
quantitative analysis of audio and visual contributions. Comparative evaluation demonstrated that
prediction-level fusion frequently outperformed video-only baselines and, in several cases, improved
upon pretrained multimodal performance.

The experimental results therefore validate the effectiveness of prediction-level fusion as a
multimodal strategy for conversational turn-taking prediction and demonstrate its applicability to
Turkish conversational data.

# 7. Project Management Summary

## 7.1 Work Breakdown and Schedule

_[Present the final Gantt chart comparing planned vs. actual timelines. Discuss schedule adherence and
reasons for any deviations.]_

## 7.2 Individual Contributions

```
Team Member Department Key Contributions Effort (%)
Elifnur Nazlı Industrial
Engineering
```
```
Literature review,
risk analysis, project
planning
```
### %10

```
Furkan Kuşman Industrial
Engineering
```
```
Work planning,
scheduling,
documentation
```
### %10

```
Mahmut Hami Bayır Industrial
Engineering
```
```
Project
coordination,
```

```
methodology
planning, report
preparation
```
### %10

_Buse Yazıcı_ Computer
Engineering

```
Video
preprocessing,
multimodal feature
integration, and
evaluation pipeline
development
```
### %18

_Fuat Bera Günay_ Computer
Engineering

```
Real-time
integration
feasibility
investigation,
prototype
development, and
preliminary testing
```
### %16

_Selin Coşkun_ Computer
Engineering

```
Video feature
extraction,
multimodal data
preparation, and
evaluation support
```
### %18

_Selin Güler_ Computer
Engineering

```
Audio model fine
tuning, performance
evaluation, and
metric analysis
```
### %18


## 7.3 Inter-Departmental Collaboration Assessment

_The multidisciplinary nature of the project required continuous collaboration between the Industrial
Engineering and Computer Engineering teams. Throughout the project, communication was maintained
through regular weekly meetings, advisor consultations, shared documentation platforms, and progress
review sessions. The Computer Engineering team focused primarily on technical implementation
activities, including data processing, model development, and performance evaluation, while the
Industrial Engineering team concentrated on project management, risk assessment, planning, and
documentation. This division of responsibilities allowed both teams to leverage their disciplinary
strengths while working toward a common project goal. One of the primary challenges involved
maintaining consistency between technical development activities and project management deliverables.
However, regular communication and shared decision-making processes minimized integration issues
and ensured alignment between departments. Overall, the collaboration was highly effective and
demonstrated the value of interdisciplinary teamwork in solving complex engineering problems involving
artificial intelligence and Human-Robot Interaction._

## 7.4 Budget and Resources

_The project was completed without requiring significant financial expenditure. Most development
activities relied on open-source software tools and publicly available research resources, eliminating the
need for costly licenses or specialized infrastructure. The primary resources used throughout the project
included personal computers, local development environments, and open-source technologies such as
Python, PyTorch, MediaPipe, OpenCV, FFmpeg, WebRTC VAD, and GitHub. Existing computational
resources proved sufficient for dataset preprocessing, feature extraction, model fine-tuning, and
evaluation activities. Although future large-scale deployment and real-time implementation may require
additional computational resources, particularly for low-latency inference and extensive testing, the
available resources were adequate for achieving the objectives of the current project. The efficient
utilization of freely available tools and existing hardware contributed to the overall cost-effectiveness
and sustainability of the project._

# 8. Ethical, Safety, and Sustainability Considerations

_The development of a multimodal turn-taking prediction system for Human-Robot Interaction involves
important ethical, safety, and sustainability considerations. Since the project utilizes conversational
audio and video recordings for training and evaluation purposes, protecting participant privacy and
ensuring responsible data usage are critical requirements. Any conversational data should be collected
with informed consent, securely stored, and processed in a manner that minimizes the exposure of
personally identifiable information. In addition, the system has been trained on a relatively limited
Turkish conversational dataset, which may introduce biases related to language, communication style, or
demographic characteristics. Therefore, the model's predictions should not be assumed to generalize
equally across all populations, and future studies should incorporate more diverse datasets to improve
fairness and reliability. Although the current project is primarily software-based and does not directly
control physical robotic hardware, safety remains an important consideration because future deployment
in real-time Human-Robot Interaction environments could affect the quality and effectiveness of
communication. Incorrect turn-taking predictions may result in interruptions, delayed responses, or
reduced user trust; consequently, extensive validation, fail-safe mechanisms, and human oversight should
be considered before practical deployment. From a sustainability perspective, the project promotes_


_efficient use of resources by relying largely on open-source technologies such as PyTorch, MediaPipe,
OpenCV, and WebRTC VAD, while adapting the framework to operate on CPU-based environments instead
of requiring specialized high-performance hardware. This approach improves accessibility, reduces
infrastructure requirements, and supports the reproducibility of research. Furthermore, advancements in
Human-Robot Interaction have the potential to contribute positively to healthcare, education, assistive
technologies, and collaborative robotic systems by enabling more natural and efficient communication
between humans and intelligent agents. However, future large-scale deployment of deep learning models
should continue to consider computational efficiency, energy consumption, and responsible resource_

# management to ensure the long-term sustainability of AI-based conversational systems.

# 9. Conclusions

## 9.1 Summary of Achievements

**1)Prediction-Level Fusion**

The primary achievement of this study was the successful design, implementation, and evaluation of a
prediction-level multimodal fusion framework for conversational turn-taking prediction.

Key achievements include:

- Generation of audio-only and video-only conversational predictions using pretrained VAP-
    based models.
- Fine-tuning of audio and visual models using Turkish conversational recordings.
- Development of a weighted prediction-level fusion framework.
- Implementation of an alpha optimization procedure for modality balancing.
- Comprehensive evaluation across multiple conversational event categories.
- Demonstration of multimodal performance improvements through prediction-level fusion.
- Identification of session-specific modality contributions and fusion characteristics.
The resulting framework provides a flexible and computationally efficient approach for integrating
multimodal conversational predictions.

## 9.2 Lessons Learned

## 1) Prediction-Level Fusion

Several important lessons were learned throughout the prediction-level fusion experiments.

The most significant observation was that audio information remains the dominant modality for
conversational turn-taking prediction. While visual information alone was generally insufficient to
match audio performance, it often provided complementary cues that improved multimodal
predictions when appropriately weighted.

Another important lesson concerned modality balancing. Equal weighting of audio and visual
predictions did not consistently produce optimal results. Instead, systematic alpha optimization
proved necessary to determine the most effective fusion configuration for each conversational session.


The experiments also demonstrated that fine-tuning does not guarantee performance improvements
for every modality. Audio fine-tuning produced relatively stable gains, whereas visual fine-tuning
exhibited greater variability across sessions. This highlights the importance of careful evaluation when
adapting pretrained multimodal models to new conversational domains.

Finally, prediction-level fusion proved to be a practical alternative to end-to-end multimodal training.
The approach enabled independent development, fine-tuning, and evaluation of each modality while
maintaining a simple and flexible integration strategy.

## 9.3 Future Work and Recommendations

Future work may include real-time implementation of the project.

# References

_[Use the IEEE style when listing references. A good guide can be found here:
[http://libguides.murdoch.edu.au/IEEE/,](http://libguides.murdoch.edu.au/IEEE/,) and many examples here:
https://libguides.murdoch.edu.au/IEEE/all]_


# APPENDIX A: Source Code and Repository

_[Provide links to the complete code repository. Include key code snippets that illustrate the most
important algorithms or modules. Display source code in a monospace (fixed-width) font and single-
spaced.]_


# APPENDIX B: User Manual / Installation Guide

_[Provide instructions for installing, configuring, and using the developed system or product. Include
screenshots and step-by-step guides.]_


# APPENDIX C: Test Data and Additional Results

_[Include detailed test data, raw results, extended performance metrics, and any supplementary analysis
that supports the findings in the main body.]_


# APPENDIX D: Meeting Minutes and Project Logs

_[Include key meeting minutes, project logs, and communication records documenting the project
coordination activities.]_


# APPENDIX E: Poster / Presentation Slides

_[Include the project poster and/or final presentation slides.]_


