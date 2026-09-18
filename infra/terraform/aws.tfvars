# Real-AWS deployment values (see ../DEPLOY-AWS.md). Used as:
#   ../.tools/terraform workspace select aws
#   ../.tools/terraform plan  -refresh=false -var-file=aws.tfvars   # dry-run
#   ../.tools/terraform apply -var-file=aws.tfvars                   # with a
#     permitted identity (the local `hackathon-deploy` user cannot — see the
#     runbook's credentials section).
#
# aws_endpoint = "" is the switch: the provider stops using dummy credentials
# and endpoint overrides and talks to real regional endpoints.

aws_endpoint = ""
region       = "us-east-1"

# `aws` slug keeps real-AWS resource names distinct from the LocalStack
# defaults (wfm-…-dev) in every console.
environment = "aws"

# S3 bucket names are global: the account id suffix is collision-proof by
# construction and self-documents ownership.
bucket_name = "wfm-raw-media-201023850731"
table_name  = "wfm-detections-aws"

# The motion Lambda ships as an ECR container image (source-built OpenCV 5
# cannot fit a 250 MB layer and needs glibc 2.39). Deterministic URI —
# scripts/push_lambda_image.py creates exactly this repo/tag. Deploy after
# it has pushed at least once.
motion_image_uri = "201023850731.dkr.ecr.us-east-1.amazonaws.com/wfm-lambda-py312-opencv5:g1"
