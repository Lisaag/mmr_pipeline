# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: streamlit/proto/NumberInput.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from streamlit.proto import LabelVisibilityMessage_pb2 as streamlit_dot_proto_dot_LabelVisibilityMessage__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!streamlit/proto/NumberInput.proto\x1a,streamlit/proto/LabelVisibilityMessage.proto\"\xbc\x03\n\x0bNumberInput\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05label\x18\x02 \x01(\t\x12\x0f\n\x07\x66orm_id\x18\x03 \x01(\t\x12\x0e\n\x06\x66ormat\x18\x08 \x01(\t\x12\x0f\n\x07has_min\x18\x0b \x01(\x08\x12\x0f\n\x07has_max\x18\x0c \x01(\x08\x12(\n\tdata_type\x18\r \x01(\x0e\x32\x15.NumberInput.DataType\x12\x14\n\x07\x64\x65\x66\x61ult\x18\x0e \x01(\x01H\x00\x88\x01\x01\x12\x0c\n\x04step\x18\x0f \x01(\x01\x12\x0b\n\x03min\x18\x10 \x01(\x01\x12\x0b\n\x03max\x18\x11 \x01(\x01\x12\x0c\n\x04help\x18\x12 \x01(\t\x12\x12\n\x05value\x18\x13 \x01(\x01H\x01\x88\x01\x01\x12\x11\n\tset_value\x18\x14 \x01(\x08\x12\x10\n\x08\x64isabled\x18\x15 \x01(\x08\x12\x31\n\x10label_visibility\x18\x16 \x01(\x0b\x32\x17.LabelVisibilityMessage\x12\x13\n\x0bplaceholder\x18\x17 \x01(\t\"\x1e\n\x08\x44\x61taType\x12\x07\n\x03INT\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x42\n\n\x08_defaultB\x08\n\x06_valueJ\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x07\x10\x08J\x04\x08\t\x10\nJ\x04\x08\n\x10\x0b\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'streamlit.proto.NumberInput_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _NUMBERINPUT._serialized_start=84
  _NUMBERINPUT._serialized_end=528
  _NUMBERINPUT_DATATYPE._serialized_start=440
  _NUMBERINPUT_DATATYPE._serialized_end=470
# @@protoc_insertion_point(module_scope)
