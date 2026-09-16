"""E2E union stage for the wildlife monitor extension repo.

Glues the sprints-1-2 inference modules (MDv6 detection + classifier v0,
both through cv2.dnn) into one reproducible pipeline stage that turns
ETL media into schema-valid detection-batch records with species labels.
"""
