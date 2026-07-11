"""
Public interface for the YOLO annotation visualization package.

The package is divided by visualization stage so dataset validation,
annotation location, and coordinate parsing remain independent and easier
to test.
"""

from .annotation_locator import (
    AnnotationPairResult,
    YoloAnnotationLocator,
)
from .annotation_parser import (
    YoloAnnotation,
    YoloAnnotationParser,
)
from .dataset_verifier import (
    DatasetVerificationResult,
    YoloDatasetVerifier,
)

from .annotation_renderer import (
    RenderedImageResult,
    YoloAnnotationRenderer,
)

__all__ = [
    "AnnotationPairResult",
    "DatasetVerificationResult",
    "RenderedImageResult",
    "YoloAnnotation",
    "YoloAnnotationLocator",
    "YoloAnnotationParser",
    "YoloAnnotationRenderer",
    "YoloDatasetVerifier",
]