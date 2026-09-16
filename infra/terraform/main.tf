provider "aws" {
  region = var.region

  # Dummy credentials: LocalStack accepts anything, and pinning them here
  # keeps `terraform apply` from touching the host credential chain.
  access_key = "test"
  secret_key = "test"

  # LocalStack is not real AWS: skip the account/credential round-trips and
  # force path-style S3 addressing (virtual-host style needs wildcard DNS).
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
  s3_use_path_style           = true

  # Every service the configuration touches, pointed at one endpoint.
  # With aws_endpoint = "" these blocks are omitted and the provider uses
  # real regional endpoints — this file then deploys unchanged to AWS.
  dynamic "endpoints" {
    for_each = var.aws_endpoint == "" ? [] : [var.aws_endpoint]
    content {
      dynamodb = endpoints.value
      s3       = endpoints.value
      lambda   = endpoints.value
      sqs      = endpoints.value
      iam      = endpoints.value
      sts      = endpoints.value
    }
  }
}

locals {
  function_name = "wfm-detection-ingest-${var.environment}"
  dlq_name      = "${local.function_name}-dlq"
  log_level     = "INFO"
}
