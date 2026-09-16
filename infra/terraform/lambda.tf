# The ingest Lambda: infra/handler/app.py + infra/schema.py zipped as-is.
# archive_file keeps `terraform apply` self-contained (no build step).

data "archive_file" "lambda" {
  type        = "zip"
  output_path = "${path.module}/../build/lambda.zip"

  source {
    content  = file("${path.module}/../handler/app.py")
    filename = "app.py"
  }
  source {
    content  = file("${path.module}/../schema.py")
    filename = "schema.py"
  }
}

resource "aws_iam_role" "ingest" {
  name = "${local.function_name}-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "ingest" {
  name = "${local.function_name}-policy"
  role = aws_iam_role.ingest.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadDetectionBatches"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:GetObjectVersion"]
        # Every object/version in the bucket; prefix filtering happens in
        # the notification config and defensively in the handler.
        Resource = ["${aws_s3_bucket.ingest.arn}/*"]
      },
      {
        Sid    = "WriteDetections"
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:TransactWriteItems",
        ]
        Resource = [
          aws_dynamodb_table.detections.arn,
          "${aws_dynamodb_table.detections.arn}/index/*",
        ]
      },
      {
        Sid    = "TailLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        # LocalStack is permissive about the account fragment; real AWS wants
        # the 12-digit account id here — one thing to revisit on first deploy.
        Resource = ["arn:aws:logs:*:*:*"]
      },
    ]
  })
}

resource "aws_lambda_function" "ingest" {
  function_name    = local.function_name
  role             = aws_iam_role.ingest.arn
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256

  handler     = "app.lambda_handler"
  runtime     = "python3.12"
  timeout     = var.lambda_timeout
  memory_size = var.lambda_memory

  environment {
    variables = merge(
      {
        DYNAMO_TABLE      = var.table_name
        DETECTIONS_PREFIX = var.detections_prefix
        MAX_DETECTIONS    = tostring(var.max_detections)
        LOG_LEVEL         = local.log_level
      },
      var.lambda_aws_endpoint == "" ? {} : { AWS_ENDPOINT_URL = var.lambda_aws_endpoint }
    )
  }

  depends_on = [aws_iam_role_policy.ingest]
}

# Async S3 invocations: 0 retries + on-failure destination = poison batches
# reach the DLQ in seconds (dev-friendly; raise retries for prod).
resource "aws_lambda_function_event_invoke_config" "ingest" {
  function_name                = aws_lambda_function.ingest.function_name
  maximum_retry_attempts       = 0
  maximum_event_age_in_seconds = 360

  destination_config {
    on_failure {
      destination = aws_sqs_queue.ingest_dlq.arn
    }
  }
}

# --- motion gate (G1) ---------------------------------------------------------
# MOG2 runs INSIDE the Lambda on the custom execution image (OpenCV 5 baked
# in — see ../docker/Dockerfile.lambda-py312-opencv5). The zip is a single
# file; cv2/numpy come from the image, so nothing heavy ships in the archive.
# It shares the ingest role: the permission surface is identical (S3 reads on
# the bucket, DynamoDB writes on the table, log tailing).

data "archive_file" "motion_lambda" {
  type        = "zip"
  output_path = "${path.module}/../build/motion-lambda.zip"

  source {
    content  = file("${path.module}/../handler/motion_app.py")
    filename = "motion_app.py"
  }
}

resource "aws_lambda_function" "motion" {
  function_name    = local.motion_function_name
  role             = aws_iam_role.ingest.arn
  filename         = data.archive_file.motion_lambda.output_path
  source_code_hash = data.archive_file.motion_lambda.output_base64sha256

  handler     = "motion_app.lambda_handler"
  runtime     = "python3.12"
  timeout     = var.motion_lambda_timeout
  memory_size = var.motion_lambda_memory

  environment {
    variables = merge(
      {
        DYNAMO_TABLE       = var.table_name
        MOTION_JOBS_PREFIX = var.motion_jobs_prefix
        MAX_JOB_FRAMES     = tostring(var.max_motion_job_frames)
        LOG_LEVEL          = local.log_level
      },
      var.lambda_aws_endpoint == "" ? {} : { AWS_ENDPOINT_URL = var.lambda_aws_endpoint }
    )
  }

  depends_on = [aws_iam_role_policy.ingest]
}

# Same poison policy as the ingest function: a malformed motion job fails
# the invocation and lands on the shared DLQ instead of retrying hot.
resource "aws_lambda_function_event_invoke_config" "motion" {
  function_name                = aws_lambda_function.motion.function_name
  maximum_retry_attempts       = 0
  maximum_event_age_in_seconds = 360

  destination_config {
    on_failure {
      destination = aws_sqs_queue.ingest_dlq.arn
    }
  }
}
