# Overridable inputs. Defaults target LocalStack (see docker-compose.yml);
# the same code deploys to real AWS by setting aws_endpoint = "" and real
# credentials via the standard AWS provider chain.

variable "aws_endpoint" {
  description = "AWS-compatible API endpoint (LocalStack edge). Empty string = real AWS. IPv6 loopback by default: docker publishes [::]:4566, and plain 127.0.0.1 is unreachable from WSL2 hosts running mirrored networking."
  type        = string
  default     = "http://[::1]:4566"
}

variable "lambda_aws_endpoint" {
  description = "AWS_ENDPOINT_URL injected into the Lambda env. Empty = let LocalStack inject its own; must stay empty on real AWS."
  type        = string
  default     = ""
}

variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment slug used in resource names (dev/stage/prod)."
  type        = string
  default     = "dev"
}

variable "bucket_name" {
  description = "Ingest bucket: raw media + detection-batch sidecars under detections/."
  type        = string
  default     = "wfm-raw-media-dev"
}

variable "table_name" {
  description = "DynamoDB single-table for normalized detections."
  type        = string
  default     = "wfm-detections-dev"
}

variable "detections_prefix" {
  description = "S3 key prefix that triggers the Lambda."
  type        = string
  default     = "detections/"
}

variable "max_detections" {
  description = "Per-batch cap enforced by the handler (defensive; S2 batches are ~1-40)."
  type        = number
  default     = 200
}

variable "lambda_timeout" {
  type    = number
  default = 30
}

variable "lambda_memory" {
  type    = number
  default = 256
}

# --- motion gate (G1: MOG2 inside the Lambda) --------------------------------

variable "motion_jobs_prefix" {
  description = "S3 prefix holding motion-job files (one JSON per camera, frames pre-sorted by capture time)."
  type        = string
  default     = "motion-jobs/"
}

variable "max_motion_job_frames" {
  description = "Per-job frame cap enforced by the handler (poison guard)."
  type        = number
  default     = 2500
}

variable "motion_lambda_timeout" {
  description = "Seconds — a job decodes up to max_motion_job_frames images end-to-end."
  type        = number
  default     = 900
}

variable "motion_lambda_memory" {
  description = "MB — image decode + MOG2 headroom (frames capped at 1024px long side)."
  type        = number
  default     = 1024
}
