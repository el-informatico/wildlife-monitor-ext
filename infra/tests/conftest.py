"""Shared fixtures: moto-mocked S3+DynamoDB stack mirroring the Terraform
table definition, plus S3-event builders."""

import json
import pathlib
import urllib.parse

import boto3
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BUCKET = "wfm-raw-media-test"
TABLE = "wfm-detections-test"


def load_fixture(name: str = "detection_batch_sample.json") -> dict:
    return json.loads((FIXTURES / name).read_text())


def create_detections_table(client, name: str = TABLE) -> None:
    """Mirror of terraform/dynamodb.tf — keep the two in sync."""
    client.create_table(
        TableName=name,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "gs1pk", "AttributeType": "S"},
            {"AttributeName": "gs1sk", "AttributeType": "S"},
            {"AttributeName": "gs2pk", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "gsi-species",
                "KeySchema": [
                    {"AttributeName": "gs1pk", "KeyType": "HASH"},
                    {"AttributeName": "gs1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "gsi-source",
                "KeySchema": [
                    {"AttributeName": "gs2pk", "KeyType": "HASH"},
                    {"AttributeName": "sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=name)


def make_event(
    key: str,
    version_id: str | None = None,
    etag: str | None = None,
    event_time: str = "2026-09-16T10:15:05.000Z",
    event_name: str = "ObjectCreated:Put",
) -> dict:
    obj: dict = {"key": urllib.parse.quote(key, safe="/"), "size": 512}
    if version_id is not None:
        obj["versionId"] = version_id
    if etag is not None:
        obj["eTag"] = etag
    return {
        "Records": [
            {
                "eventVersion": "2.2",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventName": event_name,
                "eventTime": event_time,
                "s3": {"bucket": {"name": BUCKET}, "object": obj},
            }
        ]
    }


class FakeContext:
    aws_request_id = "req-0000000000000000000000000000000"


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    """Local dummy credentials for moto; handler table override."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("DYNAMO_TABLE", TABLE)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)


@pytest.fixture
def stack():
    from moto import mock_aws

    from handler import app

    with mock_aws():
        app._reset_clients()
        dynamo = boto3.client("dynamodb", region_name="us-east-1")
        create_detections_table(dynamo)
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        # Mirrors the Terraform bucket (versioning enabled): every put_object
        # returns a distinct VersionId, like the real S3 events.
        s3.put_bucket_versioning(
            Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"}
        )
        table = boto3.resource("dynamodb", region_name="us-east-1").Table(TABLE)
        yield {"s3": s3, "dynamo": dynamo, "table": table, "app": app}
        app._reset_clients()
