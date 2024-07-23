from src.constants import DESCRIPTIONS


PLANE2COND = dict(
    zip(
        DESCRIPTIONS,
        (
            "spinal_canal_stenosis",
            "neural_foraminal_narrowing",
            "subarticular_stenosis"
        )
    )
)


CONDITIONS = [
    "spinal_canal_stenosis",
    "right_neural_foraminal_narrowing",
    "left_neural_foraminal_narrowing",
    "subarticular_stenosis"
]
