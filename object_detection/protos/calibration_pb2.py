# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/calibration.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n)object_detection/protos/calibration.proto\x12\x17object_detection.protos\"\xe4\x03\n\x11\x43'
    b'\x61librationConfig\x12P\n\x16\x66unction_approximation\x18\x01 \x01('
    b'\x0b\x32..object_detection.protos.FunctionApproximationH\x00\x12\x62\n class_id_function_approximations\x18\x02 '
    b'\x01(\x0b\x32\x36.object_detection.protos.ClassIdFunctionApproximationsH\x00\x12J\n\x13sigmoid_calibration\x18'
    b'\x03 \x01(\x0b\x32+.object_detection.protos.SigmoidCalibrationH\x00\x12\\\n\x1d\x63lass_id_sigmoid_calibrations'
    b'\x18\x04 \x01(\x0b\x32\x33.object_detection.protos.ClassIdSigmoidCalibrationsH\x00\x12\x61\n'
    b'\x1ftemperature_scaling_calibration\x18\x05 \x01('
    b'\x0b\x32\x36.object_detection.protos.TemperatureScalingCalibrationH\x00\x42\x0c\n\ncalibrator\"L\n\x15'
    b'\x46unctionApproximation\x12\x33\n\tx_y_pairs\x18\x01 \x01(\x0b\x32 '
    b'.object_detection.protos.XYPairs\"\xe9\x01\n\x1d\x43lassIdFunctionApproximations\x12l\n\x15'
    b'\x63lass_id_xy_pairs_map\x18\x01 \x03('
    b'\x0b\x32M.object_detection.protos.ClassIdFunctionApproximations.ClassIdXyPairsMapEntry\x1aZ\n\x16'
    b'\x43lassIdXyPairsMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12/\n\x05value\x18\x02 \x01(\x0b\x32 '
    b'.object_detection.protos.XYPairs:\x02\x38\x01\"\\\n\x12SigmoidCalibration\x12\x46\n\x12sigmoid_parameters\x18'
    b'\x01 \x01(\x0b\x32*.object_detection.protos.SigmoidParameters\"\x8b\x02\n\x1a\x43lassIdSigmoidCalibrations\x12'
    b'}\n\x1f\x63lass_id_sigmoid_parameters_map\x18\x01 \x03('
    b'\x0b\x32T.object_detection.protos.ClassIdSigmoidCalibrations.ClassIdSigmoidParametersMapEntry\x1an\n '
    b'ClassIdSigmoidParametersMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\x39\n\x05value\x18\x02 \x01('
    b'\x0b\x32*.object_detection.protos.SigmoidParameters:\x02\x38\x01\"/\n\x1dTemperatureScalingCalibration\x12\x0e'
    b'\n\x06scaler\x18\x01 \x01(\x02\"\xab\x01\n\x07XYPairs\x12\x39\n\x08x_y_pair\x18\x01 \x03('
    b'\x0b\x32\'.object_detection.protos.XYPairs.XYPair\x12\x45\n\x12training_data_type\x18\x02 \x01('
    b'\x0e\x32).object_detection.protos.TrainingDataType\x1a\x1e\n\x06XYPair\x12\t\n\x01x\x18\x01 \x01('
    b'\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"0\n\x11SigmoidParameters\x12\r\n\x01\x61\x18\x01 \x01('
    b'\x02:\x02-1\x12\x0c\n\x01\x62\x18\x02 \x01('
    b'\x02:\x01\x30*N\n\x10TrainingDataType\x12\x15\n\x11\x44\x41TA_TYPE_UNKNOWN\x10\x00\x12\x0f\n\x0b\x41LL_CLASSES'
    b'\x10\x01\x12\x12\n\x0e\x43LASS_SPECIFIC\x10\x02')

_TRAININGDATATYPE = DESCRIPTOR.enum_types_by_name['TrainingDataType']
TrainingDataType = enum_type_wrapper.EnumTypeWrapper(_TRAININGDATATYPE)
DATA_TYPE_UNKNOWN = 0
ALL_CLASSES = 1
CLASS_SPECIFIC = 2

_CALIBRATIONCONFIG = DESCRIPTOR.message_types_by_name['CalibrationConfig']
_FUNCTIONAPPROXIMATION = DESCRIPTOR.message_types_by_name['FunctionApproximation']
_CLASSIDFUNCTIONAPPROXIMATIONS = DESCRIPTOR.message_types_by_name['ClassIdFunctionApproximations']
_CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY = _CLASSIDFUNCTIONAPPROXIMATIONS.nested_types_by_name[
    'ClassIdXyPairsMapEntry']
_SIGMOIDCALIBRATION = DESCRIPTOR.message_types_by_name['SigmoidCalibration']
_CLASSIDSIGMOIDCALIBRATIONS = DESCRIPTOR.message_types_by_name['ClassIdSigmoidCalibrations']
_CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY = _CLASSIDSIGMOIDCALIBRATIONS.nested_types_by_name[
    'ClassIdSigmoidParametersMapEntry']
_TEMPERATURESCALINGCALIBRATION = DESCRIPTOR.message_types_by_name['TemperatureScalingCalibration']
_XYPAIRS = DESCRIPTOR.message_types_by_name['XYPairs']
_XYPAIRS_XYPAIR = _XYPAIRS.nested_types_by_name['XYPair']
_SIGMOIDPARAMETERS = DESCRIPTOR.message_types_by_name['SigmoidParameters']
CalibrationConfig = _reflection.GeneratedProtocolMessageType('CalibrationConfig', (_message.Message,), {
    'DESCRIPTOR': _CALIBRATIONCONFIG,
    '__module__': 'object_detection.protos.calibration_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.CalibrationConfig)
})
_sym_db.RegisterMessage(CalibrationConfig)

FunctionApproximation = _reflection.GeneratedProtocolMessageType('FunctionApproximation', (_message.Message,), {
    'DESCRIPTOR': _FUNCTIONAPPROXIMATION,
    '__module__': 'object_detection.protos.calibration_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.FunctionApproximation)
})
_sym_db.RegisterMessage(FunctionApproximation)

ClassIdFunctionApproximations = _reflection.GeneratedProtocolMessageType('ClassIdFunctionApproximations',
                                                                         (_message.Message,), {

                                                                             'ClassIdXyPairsMapEntry':
                                                                                 _reflection.GeneratedProtocolMessageType(
                                                                                 'ClassIdXyPairsMapEntry',
                                                                                 (_message.Message,), {
                                                                                     'DESCRIPTOR':
                                                                                         _CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY,
                                                                                     '__module__':
                                                                                         'object_detection.protos.calibration_pb2'
                                                                                     # @@protoc_insertion_point(
                                                                                         # class_scope:object_detection.protos.ClassIdFunctionApproximations.ClassIdXyPairsMapEntry)
                                                                                 })
                                                                             ,
                                                                             'DESCRIPTOR':
                                                                                 _CLASSIDFUNCTIONAPPROXIMATIONS,
                                                                             '__module__':
                                                                                 'object_detection.protos.calibration_pb2'
                                                                             # @@protoc_insertion_point(
                                                                             # class_scope:object_detection.protos.ClassIdFunctionApproximations)
                                                                         })
_sym_db.RegisterMessage(ClassIdFunctionApproximations)
_sym_db.RegisterMessage(ClassIdFunctionApproximations.ClassIdXyPairsMapEntry)

SigmoidCalibration = _reflection.GeneratedProtocolMessageType('SigmoidCalibration', (_message.Message,), {
    'DESCRIPTOR': _SIGMOIDCALIBRATION,
    '__module__': 'object_detection.protos.calibration_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.SigmoidCalibration)
})
_sym_db.RegisterMessage(SigmoidCalibration)

ClassIdSigmoidCalibrations = _reflection.GeneratedProtocolMessageType('ClassIdSigmoidCalibrations', (_message.Message,),
                                                                      {

                                                                          'ClassIdSigmoidParametersMapEntry':
                                                                              _reflection.GeneratedProtocolMessageType(
                                                                              'ClassIdSigmoidParametersMapEntry',
                                                                              (_message.Message,), {
                                                                                  'DESCRIPTOR':
                                                                                      _CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY,
                                                                                  '__module__':
                                                                                      'object_detection.protos.calibration_pb2'
                                                                                  # @@protoc_insertion_point(
                                                                                      # class_scope:object_detection.protos.ClassIdSigmoidCalibrations.ClassIdSigmoidParametersMapEntry)
                                                                              })
                                                                          ,
                                                                          'DESCRIPTOR': _CLASSIDSIGMOIDCALIBRATIONS,
                                                                          '__module__':
                                                                              'object_detection.protos.calibration_pb2'
                                                                          # @@protoc_insertion_point(
                                                                          # class_scope:object_detection.protos.ClassIdSigmoidCalibrations)
                                                                      })
_sym_db.RegisterMessage(ClassIdSigmoidCalibrations)
_sym_db.RegisterMessage(ClassIdSigmoidCalibrations.ClassIdSigmoidParametersMapEntry)

TemperatureScalingCalibration = _reflection.GeneratedProtocolMessageType('TemperatureScalingCalibration',
                                                                         (_message.Message,), {
                                                                             'DESCRIPTOR':
                                                                                 _TEMPERATURESCALINGCALIBRATION,
                                                                             '__module__':
                                                                                 'object_detection.protos.calibration_pb2'
                                                                             # @@protoc_insertion_point(
                                                                             # class_scope:object_detection.protos.TemperatureScalingCalibration)
                                                                         })
_sym_db.RegisterMessage(TemperatureScalingCalibration)

XYPairs = _reflection.GeneratedProtocolMessageType('XYPairs', (_message.Message,), {

    'XYPair': _reflection.GeneratedProtocolMessageType('XYPair', (_message.Message,), {
        'DESCRIPTOR': _XYPAIRS_XYPAIR,
        '__module__': 'object_detection.protos.calibration_pb2'
        # @@protoc_insertion_point(class_scope:object_detection.protos.XYPairs.XYPair)
    })
    ,
    'DESCRIPTOR': _XYPAIRS,
    '__module__': 'object_detection.protos.calibration_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.XYPairs)
})
_sym_db.RegisterMessage(XYPairs)
_sym_db.RegisterMessage(XYPairs.XYPair)

SigmoidParameters = _reflection.GeneratedProtocolMessageType('SigmoidParameters', (_message.Message,), {
    'DESCRIPTOR': _SIGMOIDPARAMETERS,
    '__module__': 'object_detection.protos.calibration_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.SigmoidParameters)
})
_sym_db.RegisterMessage(SigmoidParameters)

if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY._options = None
    _CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY._serialized_options = b'8\001'
    _CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY._options = None
    _CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY._serialized_options = b'8\001'
    _TRAININGDATATYPE._serialized_start = 1508
    _TRAININGDATATYPE._serialized_end = 1586
    _CALIBRATIONCONFIG._serialized_start = 71
    _CALIBRATIONCONFIG._serialized_end = 555
    _FUNCTIONAPPROXIMATION._serialized_start = 557
    _FUNCTIONAPPROXIMATION._serialized_end = 633
    _CLASSIDFUNCTIONAPPROXIMATIONS._serialized_start = 636
    _CLASSIDFUNCTIONAPPROXIMATIONS._serialized_end = 869
    _CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY._serialized_start = 779
    _CLASSIDFUNCTIONAPPROXIMATIONS_CLASSIDXYPAIRSMAPENTRY._serialized_end = 869
    _SIGMOIDCALIBRATION._serialized_start = 871
    _SIGMOIDCALIBRATION._serialized_end = 963
    _CLASSIDSIGMOIDCALIBRATIONS._serialized_start = 966
    _CLASSIDSIGMOIDCALIBRATIONS._serialized_end = 1233
    _CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY._serialized_start = 1123
    _CLASSIDSIGMOIDCALIBRATIONS_CLASSIDSIGMOIDPARAMETERSMAPENTRY._serialized_end = 1233
    _TEMPERATURESCALINGCALIBRATION._serialized_start = 1235
    _TEMPERATURESCALINGCALIBRATION._serialized_end = 1282
    _XYPAIRS._serialized_start = 1285
    _XYPAIRS._serialized_end = 1456
    _XYPAIRS_XYPAIR._serialized_start = 1426
    _XYPAIRS_XYPAIR._serialized_end = 1456
    _SIGMOIDPARAMETERS._serialized_start = 1458
    _SIGMOIDPARAMETERS._serialized_end = 1506
# @@protoc_insertion_point(module_scope)