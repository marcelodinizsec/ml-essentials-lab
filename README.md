# ml-essentials-lab
This is a curated learning space dedicated to the essential foundations of Machine Learning.

📘 Understanding ROC Curves in Machine Learning
The Receiver Operating Characteristic (ROC) Curve is one of the most important tools for evaluating binary classification models. Instead of relying on a single threshold to convert predicted probabilities into class labels, the ROC Curve shows how a model performs across all possible thresholds, providing a complete view of its ability to distinguish between positive and negative classes.

At its core, the ROC Curve plots two metrics:

True Positive Rate (TPR) – also known as Recall or Sensitivity
It measures how effectively the model identifies positive cases.

False Positive Rate (FPR)  
It measures how often the model incorrectly labels negative cases as positive.

Each point on the ROC Curve corresponds to a different decision threshold. By analyzing the curve, we can understand the trade-off between detecting true positives and avoiding false alarms. A model that achieves high TPR while maintaining low FPR is considered more effective.

A key summary metric derived from the ROC Curve is the Area Under the Curve (AUC).

An AUC close to 1.0 indicates excellent separability between classes.

An AUC of 0.5 suggests performance equivalent to random guessing.

Because ROC Curves evaluate performance independently of class imbalance and threshold selection, they are widely used in fields such as healthcare, fraud detection, and risk modeling.

All Python examples demonstrating ROC Curves—including model training, probability prediction, curve plotting, AUC calculation, and threshold analysis—are available in the repository under the src/ROC directory.