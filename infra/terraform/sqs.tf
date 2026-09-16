# Dead-letter queue for failed async invocations (poison payloads). Wired via
# the function's on-failure destination, NOT the (legacy) function-level DLQ:
# destinations survive and are visible in get_function_event_invoke_config.

resource "aws_sqs_queue" "ingest_dlq" {
  name                       = local.dlq_name
  message_retention_seconds  = 1209600 # 14 days: enough for a weekend + review
  visibility_timeout_seconds = 30      # human debug window, matches lambda timeout
}

# The Lambda service principal (not the function role) sends to the queue
# when a destination fires, so the queue needs a resource policy.
resource "aws_sqs_queue_policy" "ingest_dlq" {
  queue_url = aws_sqs_queue.ingest_dlq.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowLambdaFailureDestination"
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = "sqs:SendMessage"
        Resource  = aws_sqs_queue.ingest_dlq.arn
        Condition = {
          ArnEquals = { "AWS:SourceArn" = aws_lambda_function.ingest.arn }
        }
      }
    ]
  })
}
