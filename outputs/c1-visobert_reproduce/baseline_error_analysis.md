# Baseline Error Analysis

The weakest classes by F1 are disapproval, neutral, disappointment, nervousness, disgust.
The top co-predicted emotion pairs are disappointment + sadness, sadness + grief, anger + annoyance, amusement + joy, annoyance + disapproval, amusement + annoyance, joy + love, disgust + anger, disappointment + annoyance, optimism + caring.
These co-prediction counts indicate labels that the model often activates together, not necessarily mutually exclusive confusion errors.
Among the five lowest-F1 classes, low support appears for nervousness, which can make the classifier less stable for those emotions.
Several low-performing classes are semantically close to nearby affective states, so short Vietnamese social-media comments may not provide enough context to separate them cleanly.
High co-prediction pairs likely reflect multi-label overlap in the data as well as ambiguity between related emotions such as negative affect, uncertainty, and positive social reactions.
Rare labels and labels with subtle pragmatic cues are expected to suffer from weaker recall or precision under a fixed 0.5 threshold.
A useful next step is to inspect examples for the worst classes and test per-class thresholds or loss/augmentation strategies without changing the evaluation protocol.
