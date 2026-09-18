#!/usr/bin/env python
"""Build and push the motion-gate container image to ECR (real AWS).

Prerequisites (see ../DEPLOY-AWS.md):
  - a permitted AWS identity in the standard chain (~/.aws) — the local
    `hackathon-deploy` user has no service grants and will fail here;
  - scripts/build-lambda-image.sh has built the base tag
    wfm-lambda-py312-opencv5:g1 (LocalStack path, same OpenCV 5 runtime);
  - docker running.

What it does — all idempotent:
  1. resolve account id + region from the session (STS);
  2. create the ECR repository if missing (name wfm-lambda-py312-opencv5);
  3. build docker/Dockerfile.lambda-py312-opencv5-aws → :g1-aws
     (base image + motion_app.py + _HANDLER + RIC entrypoint);
  4. tag <acct>.dkr.ecr.<region>.amazonaws.com/wfm-lambda-py312-opencv5:g1
     and push (this URI is what terraform/aws.tfvars pins);
  5. print the image URI + digest for the deployment evidence.

Run from infra/:
    <boto3-venv>/bin/python scripts/push_lambda_image.py [--dry-run]

--dry-run stops after the STS/ECR checks + the local docker build (no
registry writes): it is the offline validation path used by DEPLOY-AWS.md.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_NAME = "wfm-lambda-py312-opencv5"
TAG = "g1"
INFRA = Path(__file__).resolve().parents[1]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=INFRA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="check perms + build locally; skip registry writes")
    args = ap.parse_args()

    session = boto3.Session()
    region = session.region_name or "us-east-1"
    sts = session.client("sts", region_name=region)
    account = sts.get_caller_identity()["Account"]
    print(f"identity: account {account} · region {region}")

    ecr = session.client("ecr", region_name=region)
    try:
        repos = ecr.describe_repositories(repositoryNames=[REPO_NAME])
        repo_uri = repos["repositories"][0]["repositoryUri"]
        print("ecr repository exists:", repo_uri)
    except ClientError as e:
        if e.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
        if args.dry_run:
            print("ecr repository missing (dry-run: not creating); "
                  "URI will be:", f"{account}.dkr.ecr.{region}.amazonaws.com/{REPO_NAME}")
            repo_uri = None
        else:
            repo_uri = ecr.create_repository(repositoryName=REPO_NAME)[
                "repository"]["repositoryUri"]
            print("ecr repository created:", repo_uri)
    if repo_uri is None:
        repo_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{REPO_NAME}"

    remote = f"{repo_uri}:{TAG}"
    sh(["docker", "build", "-f", "docker/Dockerfile.lambda-py312-opencv5-aws",
        "-t", "wfm-lambda-py312-opencv5:g1-aws", "."])

    if args.dry_run:
        print(f"dry-run: built :g1-aws locally; would tag+push {remote}")
        return 0

    token = ecr.get_authorization_token()["authorizationData"][0]
    user, _, pwd = base64.b64decode(token["authorizationToken"]).decode().partition(":")
    subprocess.run(["docker", "login", "--username", user, "--password-stdin",
                    token["proxyEndpoint"].replace("https://", "")],
                   input=pwd.encode(), check=True)

    sh(["docker", "tag", "wfm-lambda-py312-opencv5:g1-aws", remote])
    sh(["docker", "push", remote])
    digest = ecr.describe_images(repositoryName=REPO_NAME,
                                 imageIds=[{"imageTag": TAG}])["imageDetails"][0]["imageDigest"]
    print(f"\nPUSHED {remote}\ndigest {digest}")
    print("next: cd terraform && ../.tools/terraform apply -var-file=aws.tfvars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
