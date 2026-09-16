# Ingest bucket. The EC2/OpenCV node (sprint 4) uploads raw media anywhere
# under raw/ and detection-batch sidecars under detections/; only the
# detections/ prefix notifies the Lambda.

resource "aws_s3_bucket" "ingest" {
  bucket = var.bucket_name

  # Dev/LocalStack convenience: `terraform destroy` can empty the bucket.
  # For prod, remove this and use lifecycle noncurrent-version expiry.
  force_destroy = true
}

# Versioning is what makes idempotency airtight: every re-put of the same
# key yields a new versionId, and the handler's ingest markers are keyed by
# (bucket, key, versionId-or-eTag).
resource "aws_s3_bucket_versioning" "ingest" {
  bucket = aws_s3_bucket.ingest.id
  versioning_configuration {
    status = "Enabled"
  }
}

# One notification resource per bucket (AWS limit): the ingest Lambda gets
# detections/*.json, the motion Lambda gets motion-jobs/*.json. Raw media
# under raw/ notifies nobody — it is read on demand by the motion jobs.
resource "aws_s3_bucket_notification" "ingest" {
  bucket = aws_s3_bucket.ingest.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = var.detections_prefix
    filter_suffix       = ".json"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.motion.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = var.motion_jobs_prefix
    filter_suffix       = ".json"
  }

  depends_on = [aws_lambda_permission.ingest, aws_lambda_permission.motion]
}

resource "aws_lambda_permission" "ingest" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.ingest.arn
}

resource "aws_lambda_permission" "motion" {
  statement_id  = "AllowS3InvokeMotion"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.motion.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.ingest.arn
}
