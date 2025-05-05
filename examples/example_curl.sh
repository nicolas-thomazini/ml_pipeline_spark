#!/bin/bash

curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d @examples/example_request.json