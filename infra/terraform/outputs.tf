output "bucket_name" {
  value = aws_s3_bucket.ingest.bucket
}

output "table_name" {
  value = aws_dynamodb_table.detections.name
}

output "function_name" {
  value = aws_lambda_function.ingest.function_name
}

output "function_arn" {
  value = aws_lambda_function.ingest.arn
}

output "dlq_url" {
  value = aws_sqs_queue.ingest_dlq.id
}

output "dlq_arn" {
  value = aws_sqs_queue.ingest_dlq.arn
}

output "detections_prefix" {
  value = var.detections_prefix
}
