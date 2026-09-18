provider "aws" {
  region = var.region

  # LocalStack accepts any credentials, and pinning the dummy pair here keeps
  # local `terraform apply` from touching the host credential chain. On real
  # AWS (aws_endpoint = "") the nulls leave the attributes unset and the
  # provider falls back to the standard chain (~/.aws, env, roles) — see
  # DEPLOY-AWS.md.
  access_key = var.aws_endpoint == "" ? null : "test"
  secret_key = var.aws_endpoint == "" ? null : "test"

  # LocalStack is not real AWS: skip the account/credential round-trips and
  # force path-style S3 addressing (virtual-host style needs wildcard DNS).
  # On real AWS all four are false so nothing is skipped.
  skip_credentials_validation = var.aws_endpoint != ""
  skip_requesting_account_id  = var.aws_endpoint != ""
  skip_metadata_api_check     = var.aws_endpoint != ""
  s3_use_path_style           = var.aws_endpoint != ""

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
  function_name        = "wfm-detection-ingest-${var.environment}"
  motion_function_name = "wfm-motion-gate-${var.environment}"
  dlq_name             = "${local.function_name}-dlq"
  log_level            = "INFO"

  # The motion function exists either as a zip (LocalStack, where the runtime
  # image mapping supplies OpenCV) or as an ECR container image (real AWS,
  # where the source-built OpenCV 5 is baked in — a zip cannot ship it).
  # Downstream resources (permissions, notifications, invoke configs) follow
  # whichever variant is live.
  motion_use_image = var.motion_image_uri != ""
  motion_arn       = local.motion_use_image ? aws_lambda_function.motion_image[0].arn : aws_lambda_function.motion[0].arn
  motion_name      = local.motion_use_image ? aws_lambda_function.motion_image[0].function_name : aws_lambda_function.motion[0].function_name

  # The Lambda log-policy resource ARN. LocalStack accepts any account
  # fragment; real AWS scopes it to the actual account, resolved at plan time
  # (the one STS call even a locked-down deploy identity can make).
  account_id = var.aws_endpoint == "" ? data.aws_caller_identity.current[0].account_id : "*"
}

data "aws_caller_identity" "current" {
  count = var.aws_endpoint == "" ? 1 : 0
}
