# Single-table design. Keep in sync with tests/conftest.py::create_detections_table
# and the access-pattern table in infra/README.md.
#
#   detections   pk=CAM#<camera>          sk=<captured_at>|<det_id>
#   gsi-species  gs1pk=SPECIES#<sp|label> gs1sk=sk   → "what was seen"
#   gsi-source   gs2pk=SRC#<media key>    gs2sk=sk   → "what came from this image"
#   ingest mark  pk=IDEMP#<sha256>        sk=BATCH   (fast-path duplicate check)

# hash_key/range_key are deprecated upstream in favour of a key_schema block
# that provider 6.x does not implement yet; migrate on the next provider major.
resource "aws_dynamodb_table" "detections" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }
  attribute {
    name = "gs1pk"
    type = "S"
  }
  attribute {
    name = "gs1sk"
    type = "S"
  }
  attribute {
    name = "gs2pk"
    type = "S"
  }

  global_secondary_index {
    name            = "gsi-species"
    hash_key        = "gs1pk"
    range_key       = "gs1sk"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "gsi-source"
    hash_key        = "gs2pk"
    range_key       = "sk"
    projection_type = "ALL"
  }

  # Point-in-time recovery is free insurance on real AWS; LocalStack accepts
  # the flag and ignores it.
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Service     = "wildlife-monitor"
    Component   = "detection-ingest"
    Environment = var.environment
  }
}
