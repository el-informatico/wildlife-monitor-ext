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
output "motion_function_name" {
  value = local.motion_name
}
output "motion_function_arn" {
  value = local.motion_arn
}
output "motion_packaging" {
  description = "zip (LocalStack runtime mapping) or Image (ECR container, real AWS)."
  value       = local.motion_use_image ? "Image ${var.motion_image_uri}" : "zip"
}
