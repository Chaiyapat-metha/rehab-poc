from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Joint(_message.Message):
    __slots__ = ("id", "x", "y", "z", "visibility")
    ID_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    id: int
    x: float
    y: float
    z: float
    visibility: float
    def __init__(self, id: _Optional[int] = ..., x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., visibility: _Optional[float] = ...) -> None: ...

class Frame(_message.Message):
    __slots__ = ("user_id", "session_id", "frame_no", "timestamp", "joints", "features", "labels_th")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_NO_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    LABELS_TH_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    session_id: str
    frame_no: int
    timestamp: float
    joints: _containers.RepeatedCompositeFieldContainer[Joint]
    features: _containers.RepeatedScalarFieldContainer[float]
    labels_th: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, user_id: _Optional[str] = ..., session_id: _Optional[str] = ..., frame_no: _Optional[int] = ..., timestamp: _Optional[float] = ..., joints: _Optional[_Iterable[_Union[Joint, _Mapping]]] = ..., features: _Optional[_Iterable[float]] = ..., labels_th: _Optional[_Iterable[str]] = ...) -> None: ...

class Session(_message.Message):
    __slots__ = ("session_id", "user_id", "device_info", "start_ts", "end_ts", "calibration_data")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_INFO_FIELD_NUMBER: _ClassVar[int]
    START_TS_FIELD_NUMBER: _ClassVar[int]
    END_TS_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_DATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    user_id: str
    device_info: str
    start_ts: float
    end_ts: float
    calibration_data: bytes
    def __init__(self, session_id: _Optional[str] = ..., user_id: _Optional[str] = ..., device_info: _Optional[str] = ..., start_ts: _Optional[float] = ..., end_ts: _Optional[float] = ..., calibration_data: _Optional[bytes] = ...) -> None: ...

class Caption(_message.Message):
    __slots__ = ("session_id", "start_time", "end_time", "text")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    start_time: float
    end_time: float
    text: str
    def __init__(self, session_id: _Optional[str] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ..., text: _Optional[str] = ...) -> None: ...

class TrainingLabels(_message.Message):
    __slots__ = ("exercise_id", "label_class", "label_angles_vector", "label_pos_vector", "is_valid_for_training")
    EXERCISE_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_CLASS_FIELD_NUMBER: _ClassVar[int]
    LABEL_ANGLES_VECTOR_FIELD_NUMBER: _ClassVar[int]
    LABEL_POS_VECTOR_FIELD_NUMBER: _ClassVar[int]
    IS_VALID_FOR_TRAINING_FIELD_NUMBER: _ClassVar[int]
    exercise_id: str
    label_class: int
    label_angles_vector: _containers.RepeatedScalarFieldContainer[float]
    label_pos_vector: _containers.RepeatedScalarFieldContainer[float]
    is_valid_for_training: bool
    def __init__(self, exercise_id: _Optional[str] = ..., label_class: _Optional[int] = ..., label_angles_vector: _Optional[_Iterable[float]] = ..., label_pos_vector: _Optional[_Iterable[float]] = ..., is_valid_for_training: bool = ...) -> None: ...

class InferenceResult(_message.Message):
    __slots__ = ("exercise_id", "is_wrong", "class_confidence", "wrong_joints_indices", "joint_errors_mm", "angle_values", "angle_errors", "tts_text", "display_text")
    class JointErrorsMmEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    class AngleValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AngleErrorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    EXERCISE_ID_FIELD_NUMBER: _ClassVar[int]
    IS_WRONG_FIELD_NUMBER: _ClassVar[int]
    CLASS_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    WRONG_JOINTS_INDICES_FIELD_NUMBER: _ClassVar[int]
    JOINT_ERRORS_MM_FIELD_NUMBER: _ClassVar[int]
    ANGLE_VALUES_FIELD_NUMBER: _ClassVar[int]
    ANGLE_ERRORS_FIELD_NUMBER: _ClassVar[int]
    TTS_TEXT_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_TEXT_FIELD_NUMBER: _ClassVar[int]
    exercise_id: str
    is_wrong: bool
    class_confidence: float
    wrong_joints_indices: _containers.RepeatedScalarFieldContainer[int]
    joint_errors_mm: _containers.ScalarMap[int, float]
    angle_values: _containers.ScalarMap[str, float]
    angle_errors: _containers.ScalarMap[str, float]
    tts_text: str
    display_text: str
    def __init__(self, exercise_id: _Optional[str] = ..., is_wrong: bool = ..., class_confidence: _Optional[float] = ..., wrong_joints_indices: _Optional[_Iterable[int]] = ..., joint_errors_mm: _Optional[_Mapping[int, float]] = ..., angle_values: _Optional[_Mapping[str, float]] = ..., angle_errors: _Optional[_Mapping[str, float]] = ..., tts_text: _Optional[str] = ..., display_text: _Optional[str] = ...) -> None: ...
