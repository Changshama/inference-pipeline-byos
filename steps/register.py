import json

import numpy as np
import s3fs as s3fs
from sagemaker import ModelMetrics, MetricsSource
from sagemaker.s3_utils import s3_path_join
from sagemaker.serve import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.utils import unique_name_from_base


def register(
    sentence_transformer, sklearn_processor, model_approval_status, model_package_group_name, bucket
):
    model_builder = ModelBuilder(
        model=sentence_transformer,
    )
    model_metrics = ModelMetrics()
    # Notes: The register method is still under build.
    # * The registered model can not be deployed directly, which will be fixed in the next release.
    # * There will be further improvements on the register method,
    #   such as automatically filling in the content_types and response_types parameters.
    model_package = model_builder.build().register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    print(f"Registered Model Package")

    return 1