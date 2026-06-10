"""
Main features:
- Streaming preprocessing
- Streaming statistics
- Streaming metrics
- Decision tree classifier
- Bagging and Random Forest ensembles
- Stream-compatible pipeline
- Real-time metric visualisation
"""

from .io import load_csv, train_test_split, make_chunks

from .preprocessing import (
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)

from .stats import (
    RunningMean,
    RunningVariance,
    RunningHistogram,
    RunningQuantile,
    StreamingStats,
)

from .metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1Score,
    StreamingConfusionMatrix,
    RollingAccuracy,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .tree import DecisionTreeClassifier

from .ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    EnsembleClassifier,
)

from .pipeline import Pipeline
from .stream import StreamTrainer

__version__ = "1.0.0"