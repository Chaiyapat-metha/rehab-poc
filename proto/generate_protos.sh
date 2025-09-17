mkdir -p ../backend/app/proto_generated

# Compile .proto file
python -m grpc_tools.protoc \
    -I=. \
    --python_out=../backend/app/proto_generated \
    --pyi_out=../backend/app/proto_generated \
    --grpc_python_out=../backend/app/proto_generated \
    rehab.proto
touch ../backend/app/proto_generated/__init__.py